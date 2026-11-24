from typing import cast
from unittest.mock import patch

import depthai as dai
import numpy as np
import pytest

pytest.importorskip("open3d")

from depthai_nodes.message import (  # noqa: E402
    CalibrationOutcome,
    SupportPlane,
    SupportPlaneCalibrationControl,
    SupportPlaneCalibrationResult,
    SupportPlaneVerificationResult,
)
from depthai_nodes.node import (  # noqa: E402
    PlaneCalibrationConfig,
    SupportPlaneCalibrator,
)
from depthai_nodes.node.support_plane_calibrator.support_plane_calibrator import (  # noqa: E402
    IMU_MAX_AGE_SECONDS,
    MIN_ON_PLANE_POINTS,
    ON_PLANE_STRIDE,
)
from tests.utils import (  # noqa: E402
    InputMock,
    OutputMock,
    PipelineMock,
)
from tests.utils.nodes.mocks.device import (  # noqa: E402
    DeviceMock,
    camera_calibration,
    imu_calibration,
    imu_device,
)
from tests.utils.nodes.mocks.queue import OutputQueueMock  # noqa: E402
from tests.utils.scripted_open3d import ScriptedScene  # noqa: E402

CLOUD_SIZE = (20, 20)
# 100 x 100 points, enough to measure an on-plane ratio.
LARGE_CLOUD_SIZE = (100, 100)


class Source(OutputMock):
    """An upstream output; messages sent to it reach the input it is linked to.

    It runs on ``device``, or on the host when no device is given.
    """

    def __init__(self, device=None):
        super().__init__(device)
        self._target = None

    def link(self, node):
        self._target = node

    def send(self, message):
        assert self._target is not None, "source is not linked"
        self._target.send(message)


class LatestInputMock(InputMock):
    """A non-blocking input that keeps only its newest ``queueSize`` messages."""

    def __init__(self, queue_size):
        super().__init__()
        self._queue_size = queue_size

    def send(self, message):
        super().send(message)
        while len(self._queue) > self._queue_size:
            self._queue.tryGet()


class MockedCalibrator(SupportPlaneCalibrator):
    """The calibrator on mock DepthAI queues, running a set number of loop passes."""

    def __init__(self):
        self._passes = 0
        self._pipeline = PipelineMock()
        super().__init__()

    def getParentPipeline(self):
        return self._pipeline

    def createInput(self, *args, blocking=True, queueSize=None, **kwargs):
        if not blocking and queueSize is not None:
            return LatestInputMock(queueSize)
        return InputMock()

    def createOutput(self, *args, **kwargs):
        return OutputMock()

    def isRunning(self):
        self._passes -= 1
        return self._passes >= 0

    def runPasses(self, passes=1):
        self._passes = passes
        self.run()


class Harness:
    """A built calibrator.

    Its IMU source runs on device ``"1000"`` with identity IMU extrinsics, unless
    ``imus`` replaces it.
    """

    def __init__(self, config=None, size=CLOUD_SIZE, imus=None):
        self.size = size
        self.clouds = Source()
        self.imus = Source(imu_device()) if imus is None else imus
        self.node = MockedCalibrator()
        self.node.build(
            cast(dai.Node.Output, self.clouds),
            cast(dai.Node.Output, self.imus),
            config,
        )
        self.results = cast(OutputQueueMock, self.node.out.createOutputQueue())
        self.verifications = cast(
            OutputQueueMock, self.node.verificationOutput.createOutputQueue()
        )

    def control(self, control):
        self.node.inputControl.send(control)

    def start(self):
        self.control(SupportPlaneCalibrationControl.start())

    def stop(self):
        self.control(SupportPlaneCalibrationControl.stop())

    def reset(self):
        self.control(SupportPlaneCalibrationControl.reset())

    def gravity(self, z=-9.8, x=0.0, y=0.0):
        message = dai.IMUData()
        packet = dai.IMUPacket()
        packet.acceleroMeter.x = x
        packet.acceleroMeter.y = y
        packet.acceleroMeter.z = z
        message.packets = [packet]
        self.imus.send(message)

    def plane_cloud(self, depth=1000.0):
        """A flat organized cloud ``depth`` mm in front of the camera."""
        width, height = self.size
        self.depth_cloud(np.full((height, width), depth))

    def depth_cloud(self, depths):
        """An organized cloud with one ``z`` per point, rows by columns, in mm.

        Points sit on a 40 mm grid in ``x`` and ``y``; a point with ``z = 0`` has
        no depth and is sent as ``(0, 0, 0)``, as a camera sends it.
        """
        height, width = depths.shape
        points = grid_points(width, height, depths.ravel())
        points[depths.ravel() == 0] = 0
        message = dai.PointCloudData()
        message.setPoints(points.astype(np.float32))  # pyright: ignore[reportArgumentType]
        message.setWidth(width)
        message.setHeight(height)
        self.clouds.send(message)

    def unorganized_cloud(self, points):
        """A one-row cloud of ``(N, 3)`` points in mm."""
        message = dai.PointCloudData()
        message.setPoints(np.asarray(points, dtype=np.float32))  # pyright: ignore[reportArgumentType]
        message.setWidth(len(points))
        message.setHeight(1)
        self.clouds.send(message)

    def verify(self, plane):
        self.control(SupportPlaneCalibrationControl.verify(plane))

    def verification(self) -> SupportPlaneVerificationResult:
        result = self.verifications.tryGet()
        assert isinstance(result, SupportPlaneVerificationResult)
        return result

    def run(self, passes=1):
        self.node.runPasses(passes)

    def result(self) -> SupportPlaneCalibrationResult:
        result = self.results.tryGet()
        assert isinstance(result, SupportPlaneCalibrationResult)
        return result

    def no_result(self) -> bool:
        return self.results.is_empty()


def grid_points(width, height, z):
    """Points on a 40 mm grid in ``x`` and ``y``, rows by columns, with ``z``."""
    x, y = np.meshgrid(np.arange(width) * 40.0, np.arange(height) * 40.0)
    return np.column_stack([x.ravel(), y.ravel(), np.broadcast_to(z, x.size)])


@pytest.fixture
def calibrator():
    return Harness(PlaneCalibrationConfig(max_plane_passes=3))


def test_flat_cloud_succeeds_once_gravity_arrives(calibrator):
    calibrator.start()
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.no_result()

    calibrator.gravity()
    calibrator.plane_cloud()
    calibrator.run()

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert result.candidates[0].plane.offset_mm == pytest.approx(-1000, abs=1)
    assert result.candidates[0].plane.normal[2] == pytest.approx(1, abs=0.001)
    calibrator.plane_cloud(depth=1500)
    calibrator.run()
    assert calibrator.no_result()


def scripted(scene):
    return patch("open3d.geometry.PointCloud", scene.cloud)


def tilted_plane(angle_deg, offset_mm=-1000.0):
    angle = np.deg2rad(angle_deg)
    return SupportPlane(normal=(0.0, np.sin(angle), np.cos(angle)), offset_mm=offset_mm)


def depths_on(plane, size=LARGE_CLOUD_SIZE):
    """Depths that put every point of a `Harness.depth_cloud` grid on ``plane``."""
    width, height = size
    x, y = np.meshgrid(np.arange(width) * 40.0, np.arange(height) * 40.0)
    a, b, c = plane.normal
    return -(a * x + b * y + plane.offset_mm) / c


@pytest.fixture
def level_only():
    """A calibrator whose verifications judge only the level test.

    The cloud is large enough to measure an on-plane ratio, and any measured ratio is
    enough.
    """
    return Harness(
        PlaneCalibrationConfig(
            max_plane_passes=3, max_gravity_angle_deg=6.5, min_on_plane_ratio=0
        ),
        LARGE_CLOUD_SIZE,
    )


def test_cloud_sent_before_start_is_used_for_the_attempt(calibrator):
    calibrator.gravity()
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.no_result()

    calibrator.start()
    calibrator.run()

    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED


def test_clouds_without_a_plane_keep_the_attempt_searching(calibrator):
    calibrator.start()
    calibrator.gravity()
    calibrator.plane_cloud(depth=0)
    calibrator.run()
    with scripted(ScriptedScene([])):
        calibrator.plane_cloud()
        calibrator.run()
    assert calibrator.no_result()

    calibrator.plane_cloud()
    calibrator.run()

    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED


def test_tilted_planes_or_planes_above_the_camera_are_not_candidates(calibrator):
    scene = ScriptedScene(
        [
            ([0.0, 0.0, -1.0, -1000.0], 100),
            ([0.0, 0.0, 1.0, 1000.0], 100),
            ([1.0, 0.0, 0.0, -1000.0], 100),
        ]
    )
    with scripted(scene):
        calibrator.start()
        calibrator.gravity()
        calibrator.plane_cloud()
        calibrator.run()
    assert calibrator.no_result()

    calibrator.stop()
    calibrator.run()

    assert calibrator.result().outcome is CalibrationOutcome.CANCELLED


def test_floor_fitted_with_the_opposite_sign_is_a_candidate(calibrator):
    with scripted(ScriptedScene([([0.0, 0.0, -1.0, 1000.0], 100)])):
        calibrator.start()
        calibrator.gravity()
        calibrator.plane_cloud()
        calibrator.run()

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert result.candidates[0].plane == SupportPlane(normal=(0, 0, 1), offset_mm=-1000)


def calibrate(calibrator, scene):
    """Run one attempt on a scripted scene and return its result."""
    with scripted(scene):
        calibrator.start()
        calibrator.gravity()
        calibrator.plane_cloud()
        calibrator.run()
    return calibrator.result()


def floor(offset_mm):
    return SupportPlane(normal=(0, 0, 1), offset_mm=offset_mm)


def test_inlier_ratios_divide_by_the_cloud_before_planes_are_removed():
    calibrator = Harness(
        PlaneCalibrationConfig(max_plane_passes=3, min_inlier_ratio=0.2)
    )
    scene = ScriptedScene(
        [
            ([0.0, 0.0, 1.0, -1000.0], 200),
            ([0.0, 0.0, 1.0, -1500.0], 80),
            ([0.0, 0.0, 1.0, -2000.0], 79),
        ]
    )

    result = calibrate(calibrator, scene)

    # 80 of the 200 points left after the first plane is 0.4, but the ratio is
    # taken against the full preprocessed cloud.
    assert sorted((c.inlier_count, c.inlier_ratio) for c in result.candidates) == [
        (80, 0.2),
        (200, 0.5),
    ]


def test_fit_error_propagates_out_of_the_run_pass(calibrator):
    with scripted(ScriptedScene([], error=ValueError("broken geometry"))):
        calibrator.start()
        calibrator.gravity()
        calibrator.plane_cloud()
        with pytest.raises(ValueError, match="broken geometry"):
            calibrator.run()
    assert calibrator.no_result()


def test_message_that_is_not_a_point_cloud_raises_and_keeps_the_attempt(
    calibrator,
):
    calibrator.start()
    calibrator.gravity()
    calibrator.clouds.send(dai.Buffer())
    with pytest.raises(TypeError, match="PointCloudData"):
        calibrator.run()
    assert calibrator.no_result()

    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED


def test_stop_during_a_fit_cancels_after_the_fit_and_returns_to_idle(calibrator):
    # The only plane faces away from gravity, so the fit keeps searching.
    def stop_and_reset():
        calibrator.stop()
        calibrator.reset()

    scene = ScriptedScene([([0.0, 0.0, -1.0, -1000.0], 400)], during_fit=stop_and_reset)
    with scripted(scene):
        calibrator.start()
        calibrator.gravity()
        calibrator.plane_cloud()
        calibrator.run()
    assert calibrator.no_result()

    calibrator.run()

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.CANCELLED
    assert result.candidates == ()
    assert calibrator.no_result()
    calibrator.start()
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED


def test_fit_that_finds_a_plane_completes_before_stop_takes_effect(calibrator):
    scene = ScriptedScene([([0.0, 0.0, 1.0, -1000.0], 400)], during_fit=calibrator.stop)
    with scripted(scene):
        calibrator.start()
        calibrator.gravity()
        calibrator.plane_cloud()
        calibrator.run()

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert [c.plane.offset_mm for c in result.candidates] == [-1000]
    calibrator.run()
    assert calibrator.no_result()


def test_stopped_attempt_returns_to_idle_without_reset(calibrator):
    calibrator.start()
    calibrator.stop()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.CANCELLED

    calibrator.start()
    calibrator.gravity()
    calibrator.plane_cloud()
    calibrator.run()

    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED


def test_completed_attempt_ignores_start_and_stop_until_reset(calibrator):
    calibrator.start()
    calibrator.gravity()
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED

    calibrator.start()
    calibrator.stop()
    calibrator.run()
    assert calibrator.no_result()

    calibrator.reset()
    calibrator.start()
    calibrator.stop()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.CANCELLED


def test_controls_apply_in_arrival_order_and_emit_one_result(calibrator):
    calibrator.start()
    calibrator.start()
    calibrator.reset()
    calibrator.stop()
    calibrator.run()

    assert calibrator.result().outcome is CalibrationOutcome.CANCELLED
    assert calibrator.no_result()


def test_unknown_messages_on_the_control_input_are_ignored(calibrator):
    calibrator.control(dai.Buffer())
    calibrator.start()
    calibrator.stop()
    calibrator.run()

    assert calibrator.result().outcome is CalibrationOutcome.CANCELLED


def test_verification_accepts_a_plane_opposing_gravity_with_its_angle(level_only):
    plane = tilted_plane(3)
    level_only.gravity()
    level_only.verify(plane)
    level_only.plane_cloud()
    level_only.run()

    result = level_only.verification()
    assert result.plane == plane
    assert result.accepted
    assert result.gravity_angle_deg == pytest.approx(3, abs=1e-4)


def test_verification_rejects_planes_beyond_the_tolerance(level_only):
    level_only.gravity()
    level_only.verify(tilted_plane(6))
    level_only.verify(tilted_plane(7))
    level_only.plane_cloud()
    level_only.run()

    assert level_only.verification().accepted
    result = level_only.verification()
    assert not result.accepted
    assert result.gravity_angle_deg == pytest.approx(7, abs=1e-4)


def test_verification_rejects_a_plane_behind_the_camera(level_only):
    level_only.gravity()
    level_only.verify(tilted_plane(0, offset_mm=1000))
    level_only.plane_cloud()
    level_only.run()

    result = level_only.verification()
    assert not result.accepted
    assert result.gravity_angle_deg == pytest.approx(0, abs=1e-4)


def test_verification_accepts_the_opposite_sign_equation_and_echoes_it(level_only):
    plane = SupportPlane(normal=(0, 0, -1), offset_mm=1000)
    level_only.gravity()
    level_only.verify(plane)
    level_only.plane_cloud()
    level_only.run()

    result = level_only.verification()
    assert result.plane == plane
    assert result.accepted
    assert result.gravity_angle_deg == pytest.approx(0, abs=1e-4)


def test_verifications_wait_for_imu_and_are_answered_once_in_order_from_one_cloud(
    calibrator,
):
    planes = [tilted_plane(0), tilted_plane(7), tilted_plane(3)]
    for plane in planes:
        calibrator.verify(plane)
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.verifications.is_empty()

    calibrator.gravity()
    calibrator.run()
    calibrator.gravity()
    calibrator.plane_cloud()
    calibrator.run()

    results = [calibrator.verification() for _ in planes]
    assert [result.plane for result in results] == planes
    assert calibrator.verifications.is_empty()


def test_zero_accelerometer_is_not_used_for_verification(level_only):
    level_only.verify(tilted_plane(0))
    level_only.gravity(z=0)
    level_only.plane_cloud()
    level_only.run()
    assert level_only.verifications.is_empty()

    level_only.gravity()
    level_only.run()
    assert level_only.verification().accepted


def test_zero_accelerometer_is_not_used_for_fitting(calibrator):
    calibrator.start()
    calibrator.gravity(z=0)
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.no_result()

    calibrator.gravity()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED


class Clock:
    """A stand-in for ``time.monotonic`` that moves only when advanced."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    clock = Clock()
    with patch(
        "depthai_nodes.node.support_plane_calibrator.support_plane_calibrator"
        ".time.monotonic",
        clock,
    ):
        yield clock


def test_stale_imu_is_not_used_for_verification(level_only, clock):
    level_only.gravity()
    level_only.run()
    clock.now += IMU_MAX_AGE_SECONDS + 0.01

    level_only.verify(tilted_plane(0))
    level_only.plane_cloud()
    level_only.run()
    assert level_only.verifications.is_empty()

    level_only.gravity()
    level_only.run()
    assert level_only.verification().accepted


def test_stale_imu_is_not_used_for_fitting(calibrator, clock):
    calibrator.gravity()
    calibrator.run()
    clock.now += IMU_MAX_AGE_SECONDS + 0.01

    calibrator.start()
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.no_result()

    calibrator.gravity()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED


def test_imu_within_the_age_limit_is_reused(level_only, clock):
    level_only.gravity()
    level_only.run()
    clock.now += IMU_MAX_AGE_SECONDS

    level_only.verify(tilted_plane(0))
    level_only.plane_cloud()
    level_only.run()
    assert level_only.verification().accepted


def test_verification_while_idle_neither_starts_an_attempt_nor_keeps_the_cloud(
    level_only,
):
    level_only.gravity()
    level_only.verify(tilted_plane(0))
    level_only.plane_cloud()
    level_only.run()
    assert level_only.verification().accepted
    assert level_only.no_result()

    # The verification took the queued cloud, so the attempt waits for the next.
    level_only.start()
    level_only.run()
    assert level_only.no_result()

    level_only.plane_cloud()
    level_only.run()
    assert level_only.result().outcome is CalibrationOutcome.SUCCEEDED


def test_verification_while_running_is_answered_and_the_attempt_keeps_searching(
    calibrator,
):
    calibrator.start()
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    calibrator.run()
    assert calibrator.verifications.is_empty()
    assert calibrator.no_result()

    calibrator.plane_cloud(depth=0)
    calibrator.run()
    calibrator.verification()  # answered; an empty cloud can't be accepted
    assert calibrator.no_result()

    calibrator.plane_cloud()
    calibrator.run()
    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert calibrator.verifications.is_empty()


def test_one_cloud_answers_verifications_and_then_ends_the_attempt(level_only):
    level_only.start()
    level_only.gravity()
    level_only.verify(tilted_plane(0))
    level_only.verify(tilted_plane(7))
    level_only.plane_cloud()
    level_only.run()

    first, second = level_only.verification(), level_only.verification()
    assert (first.accepted, second.accepted) == (True, False)
    result = level_only.result()
    assert result.outcome is CalibrationOutcome.SUCCEEDED


def test_verification_while_complete_is_answered_without_a_new_result(level_only):
    level_only.start()
    level_only.gravity()
    level_only.plane_cloud()
    level_only.run()
    assert level_only.result().outcome is CalibrationOutcome.SUCCEEDED

    level_only.verify(tilted_plane(0))
    level_only.start()
    level_only.stop()
    level_only.run()
    assert level_only.verifications.is_empty()

    level_only.plane_cloud()
    level_only.run()
    assert level_only.verification().accepted
    assert level_only.no_result()


def test_message_that_is_not_a_point_cloud_raises_when_taken_for_a_verification(
    calibrator,
):
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    calibrator.clouds.send(dai.Buffer())
    with pytest.raises(TypeError, match="PointCloudData"):
        calibrator.run()
    assert calibrator.verifications.is_empty()


def test_verification_uses_the_built_tolerance():
    calibrator = Harness(
        PlaneCalibrationConfig(max_gravity_angle_deg=1, min_on_plane_ratio=0),
        LARGE_CLOUD_SIZE,
    )
    calibrator.gravity()
    calibrator.verify(tilted_plane(3))
    calibrator.verify(tilted_plane(0.5))
    calibrator.plane_cloud()
    calibrator.run()

    assert not calibrator.verification().accepted
    assert calibrator.verification().accepted


def test_verification_uses_the_built_camera_height_range():
    calibrator = Harness(
        PlaneCalibrationConfig(min_on_plane_ratio=0, max_camera_height_mm=1200),
        LARGE_CLOUD_SIZE,
    )
    calibrator.gravity()
    calibrator.verify(tilted_plane(0, offset_mm=-1300))
    calibrator.verify(tilted_plane(0, offset_mm=-1000))
    calibrator.plane_cloud()
    calibrator.run()

    assert not calibrator.verification().accepted
    assert calibrator.verification().accepted


def test_build_uses_the_default_config():
    calibrator = Harness(size=LARGE_CLOUD_SIZE)
    level, tilted = tilted_plane(6), tilted_plane(7)

    assert verify_on(calibrator, level, depths_on(level)).accepted
    assert not verify_on(calibrator, tilted, depths_on(tilted)).accepted


def build(config=None):
    output = cast(dai.Node.Output, OutputMock())
    return MockedCalibrator().build(output, output, config)


def test_build_rejects_a_config_of_the_wrong_type():
    with pytest.raises(TypeError):
        build(config={"max_plane_passes": 3})


def test_build_cannot_run_twice():
    node = build()
    output = cast(dai.Node.Output, OutputMock())

    with pytest.raises(RuntimeError):
        node.build(output, output)


def capped(attempts):
    return Harness(
        PlaneCalibrationConfig(max_plane_passes=3, max_calibration_attempts=attempts)
    )


def fit_empty_cloud(calibrator):
    """Send a cloud without a plane and run one pass, fitting it."""
    calibrator.plane_cloud(depth=0)
    calibrator.run()


def test_attempt_cap_ends_the_attempt_failed():
    calibrator = capped(2)
    calibrator.start()
    calibrator.gravity()
    fit_empty_cloud(calibrator)
    assert calibrator.no_result()

    fit_empty_cloud(calibrator)

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.FAILED
    assert result.candidates == ()
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.no_result()


def test_capping_cloud_with_candidates_still_succeeds():
    calibrator = capped(2)
    calibrator.start()
    calibrator.gravity()
    fit_empty_cloud(calibrator)

    calibrator.plane_cloud()
    calibrator.run()

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.SUCCEEDED


def test_clouds_overwritten_in_the_queue_do_not_count():
    calibrator = capped(2)
    calibrator.start()
    calibrator.gravity()
    calibrator.plane_cloud(depth=0)
    calibrator.plane_cloud(depth=0)
    calibrator.plane_cloud(depth=0)
    calibrator.run()
    assert calibrator.no_result()

    fit_empty_cloud(calibrator)

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.FAILED


def test_clouds_arriving_while_imu_is_stale_do_not_count(clock):
    calibrator = capped(2)
    calibrator.start()
    calibrator.gravity()
    fit_empty_cloud(calibrator)
    clock.now += IMU_MAX_AGE_SECONDS + 0.01
    fit_empty_cloud(calibrator)
    fit_empty_cloud(calibrator)
    assert calibrator.no_result()

    calibrator.gravity()
    calibrator.run()

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.FAILED


def test_clouds_taken_only_for_a_verification_do_not_count():
    calibrator = capped(1)
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    fit_empty_cloud(calibrator)
    assert calibrator.verification()

    calibrator.start()
    calibrator.run()
    assert calibrator.no_result()
    fit_empty_cloud(calibrator)

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.FAILED


def test_attempt_cap_counter_restarts_at_start_after_stop():
    calibrator = capped(2)
    calibrator.start()
    calibrator.gravity()
    fit_empty_cloud(calibrator)
    calibrator.stop()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.CANCELLED

    calibrator.start()
    fit_empty_cloud(calibrator)
    assert calibrator.no_result()
    fit_empty_cloud(calibrator)

    assert calibrator.result().outcome is CalibrationOutcome.FAILED


def test_attempt_cap_counter_restarts_at_start_after_reset():
    calibrator = capped(2)
    calibrator.start()
    calibrator.gravity()
    fit_empty_cloud(calibrator)
    fit_empty_cloud(calibrator)
    assert calibrator.result().outcome is CalibrationOutcome.FAILED

    calibrator.reset()
    calibrator.start()
    fit_empty_cloud(calibrator)
    assert calibrator.no_result()
    fit_empty_cloud(calibrator)

    assert calibrator.result().outcome is CalibrationOutcome.FAILED


def test_stop_during_the_capping_fit_reports_failed():
    calibrator = capped(1)
    # The only plane faces away from gravity, so the fit finds no candidates.
    scene = ScriptedScene(
        [([0.0, 0.0, -1.0, -1000.0], 400)], during_fit=calibrator.stop
    )
    with scripted(scene):
        calibrator.start()
        calibrator.gravity()
        calibrator.plane_cloud()
        calibrator.run()

    assert calibrator.result().outcome is CalibrationOutcome.FAILED
    calibrator.run()
    assert calibrator.no_result()


def test_attempts_without_a_cap_keep_searching(calibrator, clock):
    # Freeze IMU age so slow runners still exercise all 20 fits.
    calibrator.start()
    calibrator.gravity()
    for _ in range(20):
        fit_empty_cloud(calibrator)
    assert calibrator.no_result()

    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED


def verify_on(calibrator, plane, depths):
    calibrator.gravity()
    calibrator.verify(plane)
    calibrator.depth_cloud(depths)
    calibrator.run()
    return calibrator.verification()


@pytest.mark.parametrize(
    "offset_mm,ratio",
    [(-1000, 1.0), (-1045, 1.0), (-955, 1.0), (-1055, 0.0), (-945, 0.0)],
)
def test_on_plane_ratio_counts_points_within_the_check_distance(offset_mm, ratio):
    # 50 mm differs from the default, so the test fails if the config is ignored.
    calibrator = Harness(PlaneCalibrationConfig(check_distance_mm=50), LARGE_CLOUD_SIZE)
    floor_at_1000 = np.full((100, 100), 1000.0)

    result = verify_on(calibrator, floor(offset_mm), floor_at_1000)

    assert result.on_plane_ratio == ratio


# Sampled points on the floor at 1000 mm and on a step at 1100 mm in
# `stepped_depths`, together enough to measure a ratio.
FLOOR_SAMPLES = MIN_ON_PLANE_POINTS
STEP_SAMPLES = MIN_ON_PLANE_POINTS - 1
SAMPLES = FLOOR_SAMPLES + STEP_SAMPLES


def stepped_depths():
    """One row whose sample is `FLOOR_SAMPLES` floor points, then the step.

    The floor ends on its last sampled point, so the sampled share differs from
    the share of all points unless the sample takes 1 in ``ON_PLANE_STRIDE``.
    """
    depths = np.full((1, ON_PLANE_STRIDE * SAMPLES), 1100.0)
    depths[0, : ON_PLANE_STRIDE * (FLOOR_SAMPLES - 1) + 1] = 1000.0
    return depths


def test_on_plane_ratio_is_the_sampled_share_on_the_plane_for_either_sign():
    calibrator = Harness()
    depths = stepped_depths()

    assert (
        verify_on(calibrator, floor(-1000), depths).on_plane_ratio
        == FLOOR_SAMPLES / SAMPLES
    )
    assert verify_on(
        calibrator, SupportPlane(normal=(0, 0, -1), offset_mm=1000), depths
    ).on_plane_ratio == pytest.approx(FLOOR_SAMPLES / SAMPLES)
    assert (
        verify_on(calibrator, floor(-1100), depths).on_plane_ratio
        == STEP_SAMPLES / SAMPLES
    )


@pytest.mark.parametrize(
    ("valid_count", "measured"),
    [
        (ON_PLANE_STRIDE * (MIN_ON_PLANE_POINTS - 1) + 1, True),
        (ON_PLANE_STRIDE * (MIN_ON_PLANE_POINTS - 1), False),
    ],
    ids=["minimum-sample", "one-sample-short"],
)
@pytest.mark.parametrize("with_invalid", [False, True], ids=["dense", "interleaved"])
def test_on_plane_ratio_is_measured_from_a_minimum_sample_of_valid_points(
    valid_count, measured, with_invalid
):
    calibrator = Harness()
    # Interleaved, an invalid point off the floor, in turn the origin, NaN and
    # inf, comes before each floor point; it counts toward neither the sample
    # size nor the ratio.
    points = grid_points(valid_count, 1, 1000.0)
    if with_invalid:
        invalid = np.resize(
            [(0.0, 0.0, 0.0), (np.nan, 0.0, 1000.0), (40.0, 0.0, np.inf)],
            (valid_count, 3),
        )
        points = np.insert(points, range(valid_count), invalid, axis=0)
    calibrator.gravity()
    calibrator.verify(floor(-1000))
    calibrator.unorganized_cloud(points)
    calibrator.run()

    assert calibrator.verification().on_plane_ratio == (1.0 if measured else None)


def test_verification_rejects_a_level_plane_with_too_little_of_the_cloud_on_it():
    calibrator = Harness(
        PlaneCalibrationConfig(min_on_plane_ratio=FLOOR_SAMPLES / SAMPLES)
    )
    depths = stepped_depths()

    at_threshold = verify_on(calibrator, floor(-1000), depths)
    below = verify_on(calibrator, floor(-1100), depths)

    assert at_threshold.accepted
    assert not below.accepted


def test_verification_rejects_a_tilted_plane_the_cloud_lies_on():
    calibrator = Harness(
        PlaneCalibrationConfig(max_gravity_angle_deg=6.5), LARGE_CLOUD_SIZE
    )
    plane = tilted_plane(7)

    result = verify_on(calibrator, plane, depths_on(plane))

    assert result.on_plane_ratio == 1.0
    assert not result.accepted


def test_without_a_minimum_ratio_any_measured_level_plane_is_accepted():
    calibrator = Harness(PlaneCalibrationConfig(min_on_plane_ratio=0))
    # One floor point short of a measured ratio.
    sparse = np.full((1, ON_PLANE_STRIDE * (MIN_ON_PLANE_POINTS - 1)), 1000.0)

    off_the_cloud = verify_on(calibrator, floor(-1500), stepped_depths())
    unmeasured = verify_on(calibrator, floor(-1000), sparse)

    assert (off_the_cloud.on_plane_ratio, off_the_cloud.accepted) == (0.0, True)
    assert (unmeasured.on_plane_ratio, unmeasured.accepted) == (None, False)


def test_unorganized_cloud_with_its_floor_at_negative_z_is_fitted_and_measured():
    calibrator = Harness(PlaneCalibrationConfig(max_plane_passes=3))
    # The accelerometer reads up along +z, so the floor at z = -1000 mm is
    # 1000 mm below the camera, as another view's floor can be.
    floor_below = SupportPlane(normal=(0, 0, -1), offset_mm=-1000)
    calibrator.start()
    calibrator.gravity(z=9.8)
    calibrator.verify(floor_below)
    calibrator.unorganized_cloud(grid_points(100, 100, -1000.0))
    calibrator.run()

    assert calibrator.verification().on_plane_ratio == 1.0
    plane = calibrator.result().candidates[0].plane
    assert plane.normal == pytest.approx((0, 0, -1), abs=0.001)
    assert plane.offset_mm == pytest.approx(-1000, abs=1)


@pytest.mark.parametrize(
    ("point", "counts"),
    [
        ((0.0, 0.0, 0.0), False),
        ((np.nan, 0.0, 1000.0), False),
        ((40.0, -np.inf, 1000.0), False),
        ((40.0, 0.0, -1000.0), True),
        ((40.0, 0.0, 0.0), True),
    ],
    ids=["origin", "nan", "inf", "negative-z", "zero-z"],
)
def test_on_plane_ratio_counts_only_finite_points_other_than_the_origin(point, counts):
    calibrator = Harness()
    # Enough floor points to measure a ratio alone, after as many copies of the
    # point, which lies off the floor; counted, they take half the sample.
    floor_count = ON_PLANE_STRIDE * MIN_ON_PLANE_POINTS
    points = np.vstack(
        [np.tile(point, (floor_count, 1)), grid_points(floor_count, 1, 1000.0)]
    )
    calibrator.gravity()
    calibrator.verify(floor(-1000))
    calibrator.unorganized_cloud(points)
    calibrator.run()

    assert calibrator.verification().on_plane_ratio == (0.5 if counts else 1.0)


# The IMU's y axis is the camera's z axis: a 90° turn about x, IMU to camera.
IMU_TURNED_ABOUT_X = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def turned_imu_harness(imus):
    """A calibrator whose verifications judge only the level test, on ``imus``."""
    return Harness(
        PlaneCalibrationConfig(max_plane_passes=3, min_on_plane_ratio=0),
        LARGE_CLOUD_SIZE,
        imus,
    )


def level_floor(calibrator):
    """Fit and verify a level floor, with gravity read in the turned IMU's axes.

    Gravity of ``(0, 0, -9.8)`` in the camera's axes reads ``(0, -9.8, 0)`` on an IMU
    turned by ``IMU_TURNED_ABOUT_X``.
    """
    calibrator.start()
    calibrator.gravity(x=0.0, y=-9.8, z=0.0)
    calibrator.verify(floor(-1000))
    calibrator.plane_cloud()
    calibrator.run()
    return calibrator.result(), calibrator.verification()


def test_imu_mounted_at_90_degrees_to_the_camera_levels_the_floor():
    calibrator = turned_imu_harness(Source(imu_device(IMU_TURNED_ABOUT_X)))

    result, verification = level_floor(calibrator)

    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert verification.accepted
    assert verification.gravity_angle_deg == pytest.approx(0, abs=1e-3)


def test_imu_on_a_host_source_uses_the_default_device_calibration():
    calibrator = turned_imu_harness(Source())
    default_device = calibrator.node.getParentPipeline().getDefaultDevice()
    default_device.setCalibration(imu_calibration(IMU_TURNED_ABOUT_X))

    result, verification = level_floor(calibrator)

    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert verification.accepted
    assert verification.gravity_angle_deg == pytest.approx(0, abs=1e-3)


def test_device_without_imu_extrinsics_raises_naming_it():
    device = DeviceMock("1234")
    device.setCalibration(camera_calibration())
    calibrator = Harness(imus=Source(device))

    with pytest.raises(RuntimeError, match="1234.*IMU extrinsics") as error:
        calibrator.run()

    assert isinstance(error.value.__cause__, RuntimeError)


# Device "2000"'s local origin, CAM_B, turned 30° about y into device "1000"'s
# CAM_A, the shared origin. It does not commute with ``IMU_TURNED_ABOUT_X``.
COS_30, SIN_30 = np.sqrt(3) / 2, 0.5
CAMERA_TURNED_ABOUT_Y = np.array(
    [[COS_30, 0.0, SIN_30], [0.0, 1.0, 0.0], [-SIN_30, 0.0, COS_30]]
)


def two_device_graph():
    """A graph from device ``"2000"``'s CAM_B to device ``"1000"``'s CAM_A."""
    extrinsics = dai.Extrinsics(
        CAMERA_TURNED_ABOUT_Y.tolist(),
        dai.Point3f(10, 0, 0),
        dai.CameraBoardSocket.CAM_A,
    )
    extrinsics.toDeviceId = "1000"
    edge = dai.MultiDeviceExtrinsics()
    edge.fromDeviceId = "2000"
    edge.fromSocket = dai.CameraBoardSocket.CAM_B
    edge.extrinsics = extrinsics
    return [edge]


def imu_on_device(device_id):
    """An IMU source on ``device_id``, turned 90° about x to that device's CAM_B."""
    device = DeviceMock(device_id)
    device.setCalibration(
        imu_calibration(IMU_TURNED_ABOUT_X, dai.CameraBoardSocket.CAM_B)
    )
    return Source(device)


@pytest.mark.parametrize("set_after_build", [False, True])
def test_imu_on_the_non_reference_device_levels_the_floor_in_shared_coordinates(
    set_after_build,
):
    pipeline = PipelineMock()
    if not set_after_build:
        pipeline.setMultiDeviceCalibration(two_device_graph())
    with patch.object(MockedCalibrator, "getParentPipeline", return_value=pipeline):
        calibrator = turned_imu_harness(imu_on_device("2000"))
        if set_after_build:
            pipeline.setMultiDeviceCalibration(two_device_graph())
        calibrator.start()
        # Gravity of (0, 0, -9.8) in shared coordinates, read on the IMU.
        calibrator.gravity(x=9.8 * SIN_30, y=-9.8 * COS_30, z=0.0)
        calibrator.verify(floor(-1000))
        calibrator.plane_cloud()
        calibrator.run()

    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED
    verification = calibrator.verification()
    assert verification.accepted
    assert verification.gravity_angle_deg == pytest.approx(0, abs=1e-3)


def test_imu_device_missing_from_the_multi_device_calibration_raises_naming_it():
    pipeline = PipelineMock()
    pipeline.setMultiDeviceCalibration(two_device_graph())
    with patch.object(MockedCalibrator, "getParentPipeline", return_value=pipeline):
        calibrator = Harness(imus=Source(imu_device(deviceId="3000")))

        with pytest.raises(RuntimeError, match="3000.*multi-device calibration"):
            calibrator.run()
