from datetime import timedelta
from typing import NamedTuple, cast
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
    SupportPlaneCandidate,
    SupportPlaneVerificationResult,
)
from depthai_nodes.node import (  # noqa: E402
    PlaneCalibrationConfig,
    SupportPlaneCalibrator,
)
from depthai_nodes.node.support_plane_calibrator.support_plane_calibrator import (  # noqa: E402
    IMU_MAX_AGE_SECONDS,
)
from tests.utils import (  # noqa: E402
    InputMock,
    OutputMock,
    PipelineMock,
)
from tests.utils.nodes.mocks.queue import OutputQueueMock  # noqa: E402
from tests.utils.scripted_open3d import ScriptedScene  # noqa: E402

CLOUD_SIZE = (20, 20)


class Stamps(NamedTuple):
    """The DepthAI stamps a message carries."""

    timestamp: timedelta
    timestamp_device: timedelta
    sequence_num: int


UNSET_STAMPS = Stamps(timedelta(0), timedelta(0), 0)


def stamps(sequence_num):
    """Distinct, nonzero stamps keyed by a sequence number."""
    return Stamps(
        timedelta(seconds=100 + sequence_num),
        timedelta(seconds=50 + sequence_num),
        sequence_num,
    )


def stamps_of(message):
    return Stamps(
        message.getTimestamp(), message.getTimestampDevice(), message.getSequenceNum()
    )


class Source(OutputMock):
    """An upstream output; messages sent to it reach the input it is linked to."""

    def __init__(self):
        super().__init__()
        self._target = None

    def link(self, node):
        self._target = node

    def send(self, message):
        assert self._target is not None, "source is not linked"
        self._target.send(message)


class MockedCalibrator(SupportPlaneCalibrator):
    """The calibrator on mock DepthAI queues, running a set number of loop passes."""

    def __init__(self):
        self._passes = 0
        self._pipeline = PipelineMock()
        super().__init__()

    def getParentPipeline(self):
        return self._pipeline

    def createInput(self, *args, **kwargs):
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
    def __init__(self, config=None):
        self.clouds = Source()
        self.imus = Source()
        self.node = MockedCalibrator()
        self.node.build(
            cast(dai.Node.Output, self.clouds),
            cast(dai.Node.Output, self.imus),
            CLOUD_SIZE,
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

    def gravity(self, z=-9.8):
        message = dai.IMUData()
        packet = dai.IMUPacket()
        packet.acceleroMeter.x = 0
        packet.acceleroMeter.y = 0
        packet.acceleroMeter.z = z
        message.packets = [packet]
        self.imus.send(message)

    def plane_cloud(self, depth=1000.0, stamps: Stamps | None = None):
        """A flat organized cloud ``depth`` mm in front of the camera."""
        x, y = np.meshgrid(np.arange(20) * 40.0, np.arange(20) * 40.0)
        points = np.column_stack([x.ravel(), y.ravel(), np.full(400, depth)])
        message = dai.PointCloudData()
        message.setPoints(points.astype(np.float32))  # pyright: ignore[reportArgumentType]
        if stamps is not None:
            message.setTimestamp(stamps.timestamp)
            message.setTimestampDevice(stamps.timestamp_device)
            message.setSequenceNum(stamps.sequence_num)
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


def test_succeeded_result_carries_the_stamps_of_the_cloud_that_yielded_candidates(
    calibrator,
):
    calibrator.start()
    calibrator.gravity()
    calibrator.plane_cloud(depth=0, stamps=stamps(1))
    calibrator.run()
    calibrator.plane_cloud(stamps=stamps(2))
    calibrator.run()

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert stamps_of(result) == stamps(2)


def test_cancelled_result_carries_the_stamps_of_the_last_fitted_cloud(calibrator):
    calibrator.start()
    calibrator.gravity()
    calibrator.plane_cloud(depth=0, stamps=stamps(1))
    calibrator.run()
    calibrator.plane_cloud(depth=0, stamps=stamps(2))
    calibrator.run()

    calibrator.stop()
    calibrator.run()

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.CANCELLED
    assert stamps_of(result) == stamps(2)


def test_cancelled_result_before_any_fit_keeps_unset_stamps(calibrator):
    calibrator.start()
    calibrator.gravity()
    calibrator.plane_cloud(depth=0, stamps=stamps(1))
    calibrator.run()
    calibrator.stop()
    calibrator.run()
    assert stamps_of(calibrator.result()) == stamps(1)

    calibrator.start()
    calibrator.stop()
    calibrator.run()

    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.CANCELLED
    assert stamps_of(result) == UNSET_STAMPS


def scripted(scene):
    return patch("open3d.geometry.PointCloud", scene.cloud)


def tilted_plane(angle_deg, offset_mm=-1000.0):
    angle = np.deg2rad(angle_deg)
    return SupportPlane(normal=(0.0, np.sin(angle), np.cos(angle)), offset_mm=offset_mm)


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


def test_candidates_carry_their_plane_inlier_count_and_ratio(calibrator):
    # The scripted preprocessed cloud has 400 points.
    result = calibrate(calibrator, ScriptedScene([([0.0, 0.0, 1.0, -1000.0], 120)]))

    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert result.candidates == (
        SupportPlaneCandidate(plane=floor(-1000), inlier_count=120, inlier_ratio=0.3),
    )


def test_candidates_are_ordered_by_inliers_then_by_nearest_offset(calibrator):
    scene = ScriptedScene(
        [
            ([0.0, 0.0, 1.0, -1500.0], 100),
            ([0.0, 0.0, 1.0, -2000.0], 150),
            ([0.0, 0.0, 1.0, -1000.0], 100),
        ]
    )

    result = calibrate(calibrator, scene)

    assert [
        (c.plane.offset_mm, c.inlier_count, c.inlier_ratio) for c in result.candidates
    ] == [(-2000, 150, 0.375), (-1000, 100, 0.25), (-1500, 100, 0.25)]


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
    assert [(c.inlier_count, c.inlier_ratio) for c in result.candidates] == [
        (200, 0.5),
        (80, 0.2),
    ]
    assert all(c.inlier_ratio >= 0.2 for c in result.candidates)


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


def test_cloud_of_the_wrong_size_raises_and_keeps_the_attempt(calibrator):
    calibrator.start()
    calibrator.gravity()
    cloud = dai.PointCloudData()
    cloud.setPoints(np.ones((10, 3), dtype=np.float32))  # pyright: ignore[reportArgumentType]
    calibrator.clouds.send(cloud)
    with pytest.raises(ValueError, match="organized"):
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


def test_verification_accepts_a_plane_opposing_gravity_with_its_angle(calibrator):
    plane = tilted_plane(3)
    calibrator.gravity()
    calibrator.verify(plane)
    calibrator.plane_cloud()
    calibrator.run()

    result = calibrator.verification()
    assert result.plane == plane
    assert result.accepted
    assert result.gravity_angle_deg == pytest.approx(3, abs=1e-4)


def test_verification_rejects_planes_beyond_the_tolerance(calibrator):
    calibrator.gravity()
    calibrator.verify(tilted_plane(6))
    calibrator.verify(tilted_plane(7))
    calibrator.plane_cloud()
    calibrator.run()

    assert calibrator.verification().accepted
    result = calibrator.verification()
    assert not result.accepted
    assert result.gravity_angle_deg == pytest.approx(7, abs=1e-4)


def test_verification_rejects_a_plane_behind_the_camera(calibrator):
    calibrator.gravity()
    calibrator.verify(tilted_plane(0, offset_mm=1000))
    calibrator.plane_cloud()
    calibrator.run()

    result = calibrator.verification()
    assert not result.accepted
    assert result.gravity_angle_deg == pytest.approx(0, abs=1e-4)


def test_verification_accepts_the_opposite_sign_equation_and_echoes_it(calibrator):
    plane = SupportPlane(normal=(0, 0, -1), offset_mm=1000)
    calibrator.gravity()
    calibrator.verify(plane)
    calibrator.plane_cloud()
    calibrator.run()

    result = calibrator.verification()
    assert result.plane == plane
    assert result.accepted
    assert result.gravity_angle_deg == pytest.approx(0, abs=1e-4)


def test_verifications_wait_for_imu_and_are_answered_once_in_order_from_one_cloud(
    calibrator,
):
    planes = [tilted_plane(0), tilted_plane(7), tilted_plane(3)]
    for plane in planes:
        calibrator.verify(plane)
    calibrator.plane_cloud(stamps=stamps(1))
    calibrator.run()
    assert calibrator.verifications.is_empty()

    calibrator.gravity()
    calibrator.run()
    calibrator.gravity()
    calibrator.plane_cloud(stamps=stamps(2))
    calibrator.run()

    results = [calibrator.verification() for _ in planes]
    assert [result.plane for result in results] == planes
    assert [stamps_of(result) for result in results] == [stamps(1)] * 3
    assert calibrator.verifications.is_empty()


def test_zero_accelerometer_is_not_used_for_verification(calibrator):
    calibrator.verify(tilted_plane(0))
    calibrator.gravity(z=0)
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.verifications.is_empty()

    calibrator.gravity()
    calibrator.run()
    assert calibrator.verification().accepted


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


def test_stale_imu_is_not_used_for_verification(calibrator, clock):
    calibrator.gravity()
    calibrator.run()
    clock.now += IMU_MAX_AGE_SECONDS + 0.01

    calibrator.verify(tilted_plane(0))
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.verifications.is_empty()

    calibrator.gravity()
    calibrator.run()
    assert calibrator.verification().accepted


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


def test_imu_within_the_age_limit_is_reused(calibrator, clock):
    calibrator.gravity()
    calibrator.run()
    clock.now += IMU_MAX_AGE_SECONDS

    calibrator.verify(tilted_plane(0))
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.verification().accepted


def test_verification_while_idle_neither_starts_an_attempt_nor_keeps_the_cloud(
    calibrator,
):
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.verification().accepted
    assert calibrator.no_result()

    # The verification took the queued cloud, so the attempt waits for the next.
    calibrator.start()
    calibrator.run()
    assert calibrator.no_result()

    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED


def test_verification_while_running_is_answered_and_the_attempt_keeps_searching(
    calibrator,
):
    calibrator.start()
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    calibrator.run()
    assert calibrator.verifications.is_empty()
    assert calibrator.no_result()

    calibrator.plane_cloud(depth=0, stamps=stamps(4))
    calibrator.run()
    assert stamps_of(calibrator.verification()) == stamps(4)
    assert calibrator.no_result()

    calibrator.plane_cloud(stamps=stamps(5))
    calibrator.run()
    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert stamps_of(result) == stamps(5)
    assert calibrator.verifications.is_empty()


def test_one_cloud_answers_verifications_and_then_ends_the_attempt(calibrator):
    calibrator.start()
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    calibrator.verify(tilted_plane(7))
    calibrator.plane_cloud(stamps=stamps(3))
    calibrator.run()

    first, second = calibrator.verification(), calibrator.verification()
    assert (first.accepted, second.accepted) == (True, False)
    assert stamps_of(first) == stamps_of(second) == stamps(3)
    result = calibrator.result()
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert stamps_of(result) == stamps(3)


def test_verification_while_complete_is_answered_without_a_new_result(calibrator):
    calibrator.start()
    calibrator.gravity()
    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED

    calibrator.verify(tilted_plane(0))
    calibrator.start()
    calibrator.stop()
    calibrator.run()
    assert calibrator.verifications.is_empty()

    calibrator.plane_cloud(stamps=stamps(6))
    calibrator.run()
    assert stamps_of(calibrator.verification()) == stamps(6)
    assert calibrator.no_result()


def test_message_that_is_not_a_point_cloud_raises_when_taken_for_a_verification(
    calibrator,
):
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    calibrator.clouds.send(dai.Buffer())
    with pytest.raises(TypeError, match="PointCloudData"):
        calibrator.run()
    assert calibrator.verifications.is_empty()


def test_cloud_of_the_wrong_size_raises_when_taken_for_a_verification(calibrator):
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    cloud = dai.PointCloudData()
    cloud.setPoints(np.ones((10, 3), dtype=np.float32))  # pyright: ignore[reportArgumentType]
    calibrator.clouds.send(cloud)
    with pytest.raises(ValueError, match="organized"):
        calibrator.run()
    assert calibrator.verifications.is_empty()


def test_verification_uses_the_built_tolerance():
    calibrator = Harness(PlaneCalibrationConfig(max_gravity_angle_deg=1))
    calibrator.gravity()
    calibrator.verify(tilted_plane(3))
    calibrator.verify(tilted_plane(0.5))
    calibrator.plane_cloud()
    calibrator.run()

    assert not calibrator.verification().accepted
    assert calibrator.verification().accepted


def test_build_uses_the_default_config():
    calibrator = Harness()
    calibrator.gravity()
    calibrator.verify(tilted_plane(6))
    calibrator.verify(tilted_plane(7))
    calibrator.plane_cloud()
    calibrator.run()

    assert calibrator.verification().accepted
    assert not calibrator.verification().accepted


def build(size=CLOUD_SIZE, config=None):
    output = cast(dai.Node.Output, OutputMock())
    return MockedCalibrator().build(output, output, size, config)


@pytest.mark.parametrize("size", [(0, 48), (64, -1), (64,), (1, 2, 3), [64]])
def test_build_rejects_invalid_point_cloud_sizes(size):
    with pytest.raises(ValueError):
        build(size)


@pytest.mark.parametrize(
    "size", [(64.0, 48), (True, True), ("64", 48), [64, True], None, 64, b"ab"]
)
def test_build_rejects_point_cloud_sizes_of_the_wrong_type(size):
    with pytest.raises(TypeError):
        build(size)


def test_build_accepts_a_point_cloud_size_list():
    build([20, 20])


def test_build_accepts_numpy_integer_sizes():
    build((np.int64(20), np.int32(20)))


def test_build_rejects_a_config_of_the_wrong_type():
    with pytest.raises(TypeError):
        build(config={"max_plane_passes": 3})


def test_build_cannot_run_twice():
    node = build()
    output = cast(dai.Node.Output, OutputMock())

    with pytest.raises(RuntimeError):
        node.build(output, output, CLOUD_SIZE)


def test_verification_while_idle_waits_for_a_cloud_and_carries_its_stamps(
    calibrator,
):
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    calibrator.run()
    assert calibrator.verifications.is_empty()

    calibrator.plane_cloud(stamps=stamps(7))
    calibrator.run()

    result = calibrator.verification()
    assert result.accepted
    assert stamps_of(result) == stamps(7)
    assert calibrator.verifications.is_empty()
