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
)
from tests.utils import (  # noqa: E402
    InputMock,
    OutputMock,
    PipelineMock,
)
from tests.utils.nodes.mocks.queue import OutputQueueMock  # noqa: E402
from tests.utils.scripted_open3d import ScriptedScene  # noqa: E402

CLOUD_SIZE = (20, 20)


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

    def plane_cloud(self, depth=1000.0):
        """A flat organized cloud ``depth`` mm in front of the camera."""
        x, y = np.meshgrid(np.arange(20) * 40.0, np.arange(20) * 40.0)
        points = np.column_stack([x.ravel(), y.ravel(), np.full(400, depth)])
        message = dai.PointCloudData()
        message.setPoints(points.astype(np.float32))  # pyright: ignore[reportArgumentType]
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
    assert result.candidates[0].offset_mm == pytest.approx(-1000, abs=1)
    assert result.candidates[0].normal[2] == pytest.approx(1, abs=0.001)
    calibrator.plane_cloud(depth=1500)
    calibrator.run()
    assert calibrator.no_result()


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
    assert result.candidates[0] == SupportPlane(normal=(0, 0, 1), offset_mm=-1000)


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
    assert [p.offset_mm for p in result.candidates] == [-1000]
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
    calibrator.run()

    result = calibrator.verification()
    assert result.plane == plane
    assert result.accepted
    assert result.gravity_angle_deg == pytest.approx(3, abs=1e-4)


def test_verification_rejects_planes_beyond_the_tolerance(calibrator):
    calibrator.gravity()
    calibrator.verify(tilted_plane(6))
    calibrator.verify(tilted_plane(7))
    calibrator.run()

    assert calibrator.verification().accepted
    result = calibrator.verification()
    assert not result.accepted
    assert result.gravity_angle_deg == pytest.approx(7, abs=1e-4)


def test_verification_rejects_a_plane_behind_the_camera(calibrator):
    calibrator.gravity()
    calibrator.verify(tilted_plane(0, offset_mm=1000))
    calibrator.run()

    result = calibrator.verification()
    assert not result.accepted
    assert result.gravity_angle_deg == pytest.approx(0, abs=1e-4)


def test_verification_accepts_the_opposite_sign_equation_and_echoes_it(calibrator):
    plane = SupportPlane(normal=(0, 0, -1), offset_mm=1000)
    calibrator.gravity()
    calibrator.verify(plane)
    calibrator.run()

    result = calibrator.verification()
    assert result.plane == plane
    assert result.accepted
    assert result.gravity_angle_deg == pytest.approx(0, abs=1e-4)


def test_verifications_wait_for_imu_and_are_answered_once_in_order(calibrator):
    planes = [tilted_plane(0), tilted_plane(7), tilted_plane(3)]
    for plane in planes:
        calibrator.verify(plane)
    calibrator.run()
    assert calibrator.verifications.is_empty()

    calibrator.gravity()
    calibrator.run()
    calibrator.gravity()
    calibrator.run()

    assert [calibrator.verification().plane for _ in planes] == planes
    assert calibrator.verifications.is_empty()


def test_zero_accelerometer_is_not_used_for_verification(calibrator):
    calibrator.verify(tilted_plane(0))
    calibrator.gravity(z=0)
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
    calibrator.run()
    assert calibrator.verification().accepted


def test_verification_does_not_affect_attempts(calibrator):
    calibrator.start()
    calibrator.gravity()
    calibrator.verify(tilted_plane(0))
    calibrator.run()
    assert calibrator.verification().accepted
    assert calibrator.no_result()

    calibrator.plane_cloud()
    calibrator.run()
    assert calibrator.result().outcome is CalibrationOutcome.SUCCEEDED

    calibrator.verify(tilted_plane(0))
    calibrator.start()
    calibrator.stop()
    calibrator.run()
    assert calibrator.verification().accepted
    assert calibrator.no_result()


def test_verification_uses_the_built_tolerance():
    calibrator = Harness(PlaneCalibrationConfig(max_gravity_angle_deg=1))
    calibrator.gravity()
    calibrator.verify(tilted_plane(3))
    calibrator.verify(tilted_plane(0.5))
    calibrator.run()

    assert not calibrator.verification().accepted
    assert calibrator.verification().accepted


def test_build_uses_the_default_config():
    calibrator = Harness()
    calibrator.gravity()
    calibrator.verify(tilted_plane(6))
    calibrator.verify(tilted_plane(7))
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
