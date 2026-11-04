import time
from collections import deque
from enum import Enum
from typing import List, Optional, Tuple

import depthai as dai
import numpy as np

from depthai_nodes.message.support_plane import SupportPlane
from depthai_nodes.message.support_plane_calibration import (
    CalibrationOutcome,
    PlaneCalibrationConfig,
    SupportPlaneCalibrationControl,
    SupportPlaneCalibrationResult,
    SupportPlaneVerificationResult,
)
from depthai_nodes.message.utils.value_checks import check_instance, check_int
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.utils.open3d_dependency import require_open3d

from .plane_validation import gravity_angle_deg, is_valid_horizontal_plane
from .support_plane_fitter import SupportPlaneFitter

# Pause between run-loop passes, as DepthAI's DynamicCalibration does.
PASS_SLEEP_SECONDS = 0.005


class CalibratorState(Enum):
    """Where the calibrator is in an attempt; see `SupportPlaneCalibrator`."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"


class SupportPlaneCalibrator(BaseThreadedHostNode):
    """Find the support plane in single point clouds, checked against gravity.

    Send `depthai_nodes.message.SupportPlaneCalibrationControl` messages to
    ``inputControl``. Each attempt ends with one
    `depthai_nodes.message.SupportPlaneCalibrationResult` on ``out``, and each
    verify command produces one
    `depthai_nodes.message.SupportPlaneVerificationResult` on
    ``verificationOutput``. Requires open3d, installed with
    ``pip install depthai-nodes[box-measurement]``.

    States:
        - Idle: Start begins an attempt; Stop and Reset do nothing.
        - Running: each loop pass fits the latest point cloud once an IMU
          sample is available, until a cloud yields candidates (succeeded), a
          fit fails (failed), or Stop arrives (cancelled). Start is ignored.
        - Complete: after success or failure; Reset returns to idle, and Start
          and Stop are ignored until then. A cancelled attempt returns straight
          to idle without Reset.

    Each loop pass applies pending controls in arrival order, takes the latest
    IMU sample, answers verifications and, while running, fits the latest
    cloud. Controls wait during a fit, so Stop takes effect after the current
    fit: a fit that completes the attempt reports its outcome, otherwise the
    attempt ends cancelled.

    Verification is independent of attempts: verify commands are answered in
    order once an IMU sample is available.

    Errors: a non-``PointCloudData`` cloud, or a fit that raises
    ``ValueError`` or ``RuntimeError``, fails the attempt. Any other error is a
    bug and is not caught: DepthAI prints it with the raising line and aborts
    the process. Outputs use standard DepthAI buffering and backpressure.
    """

    def __init__(self) -> None:
        """Create the control, point cloud and IMU inputs and the outputs."""
        super().__init__()
        self._control_input = self.createInput(name="control")
        self._out = self.createOutput()
        self._verification_output = self.createOutput(name="verification")
        # Keep only the latest cloud, so idle never backpressures the shared
        # stream and a cloud arriving just before Start is still there for it.
        self._point_cloud_input = self.createInput(
            name="pointCloud", blocking=False, queueSize=1
        )
        self._imu_input = self.createInput(name="imu")
        self._state = CalibratorState.IDLE
        self._config: Optional[PlaneCalibrationConfig] = None
        self._fitter: Optional[SupportPlaneFitter] = None
        self._verifications: deque[SupportPlane] = deque()
        self._latest_imu: Optional[dai.IMUData] = None

    @property
    def inputControl(self) -> dai.Node.Input:
        """Input for `depthai_nodes.message.SupportPlaneCalibrationControl`.

        Returns:
            ``dai.Node.Input``: The control input.
        """
        return self._control_input

    @property
    def out(self) -> dai.Node.Output:
        """One `depthai_nodes.message.SupportPlaneCalibrationResult` per attempt.

        Returns:
            ``dai.Node.Output``: The calibration result output.
        """
        return self._out

    @property
    def verificationOutput(self) -> dai.Node.Output:
        """One `depthai_nodes.message.SupportPlaneVerificationResult` per verify.

        Returns:
            ``dai.Node.Output``: The verification result output.
        """
        return self._verification_output

    def build(
        self,
        pointCloud: dai.Node.Output,
        imu: dai.Node.Output,
        pointCloudSize: Tuple[int, int],
        config: Optional[PlaneCalibrationConfig] = None,
    ) -> "SupportPlaneCalibrator":
        """Link the inputs and fix the cloud size and fitting settings.

        Args:
            pointCloud: Output emitting organized ``dai.PointCloudData`` in
                millimeters.
            imu: Output emitting ``dai.IMUData`` with accelerometer packets.
            pointCloudSize: Cloud ``(width, height)`` in points.
            config: Plane-fitting settings; ``None`` uses the defaults.

        Returns:
            ``SupportPlaneCalibrator``: The built node.

        Raises:
            ImportError: If open3d is not installed.
            RuntimeError: If the node is already built.
            TypeError: If ``pointCloudSize`` is not a pair of integers or
                ``config`` is not a ``PlaneCalibrationConfig``.
            ValueError: If ``pointCloudSize`` does not have two positive values.
        """
        require_open3d(type(self).__name__)
        if self._fitter is not None:
            raise RuntimeError(
                "SupportPlaneCalibrator is already built; create a new node to "
                "change its configuration"
            )
        check_instance("pointCloudSize", pointCloudSize, tuple)
        if len(pointCloudSize) != 2:
            raise ValueError("pointCloudSize must be (width, height)")
        width = check_int("pointCloudSize width", pointCloudSize[0])
        height = check_int("pointCloudSize height", pointCloudSize[1])
        if width <= 0 or height <= 0:
            raise ValueError(
                f"pointCloudSize must be positive, got ({width}, {height})"
            )
        if config is None:
            config = PlaneCalibrationConfig()
        check_instance("config", config, PlaneCalibrationConfig)
        self._config = config
        self._fitter = SupportPlaneFitter((width, height), config)
        pointCloud.link(self._point_cloud_input)
        imu.link(self._imu_input)
        return self

    def run(self) -> None:
        """Apply controls, answer verifications and fit clouds until stopped."""
        try:
            while self.isRunning():
                self._apply_controls()
                self._take_latest_imu()
                imu = self._latest_imu
                if imu is not None:
                    self._answer_verifications(imu)
                if self._state is CalibratorState.RUNNING and imu is not None:
                    cloud = self._point_cloud_input.tryGet()
                    if cloud is not None:
                        self._process_cloud(cloud, imu)
                time.sleep(PASS_SLEEP_SECONDS)
        # DepthAI's stub declares QueueException through an undeclared nested
        # class, so pyright cannot tell it is an exception; at runtime it is one.
        except dai.MessageQueue.QueueException:  # pyright: ignore[reportGeneralTypeIssues]
            return

    def _apply_controls(self) -> None:
        """Apply every pending control in arrival order."""
        commands = SupportPlaneCalibrationControl.Commands
        while True:
            control = self._control_input.tryGet()
            if control is None:
                return
            if not isinstance(control, SupportPlaneCalibrationControl):
                continue
            command = control.command
            if isinstance(command, commands.Verify):
                self._verifications.append(command.plane)
            elif isinstance(command, commands.Start):
                if self._state is CalibratorState.IDLE:
                    self._state = CalibratorState.RUNNING
            elif isinstance(command, commands.Stop):
                if self._state is CalibratorState.RUNNING:
                    self._finish(
                        CalibrationOutcome.CANCELLED, [], "Calibration cancelled"
                    )
            elif self._state is not CalibratorState.RUNNING:
                self._state = CalibratorState.IDLE

    def _take_latest_imu(self) -> None:
        while True:
            imu = self._imu_input.tryGet()
            if imu is None:
                return
            if isinstance(imu, dai.IMUData):
                self._latest_imu = imu

    def _finish(
        self, outcome: CalibrationOutcome, candidates: List[SupportPlane], info: str
    ) -> None:
        # Like DepthAI MultiDeviceCalibration, a stopped attempt needs no reset.
        self._state = (
            CalibratorState.IDLE
            if outcome is CalibrationOutcome.CANCELLED
            else CalibratorState.COMPLETE
        )
        self._out.send(SupportPlaneCalibrationResult(outcome, candidates, info))

    def _answer_verifications(self, imu: dai.IMUData) -> None:
        if not self._verifications:
            return
        gravity = _gravity(imu)
        if gravity is None:
            return
        assert self._config is not None
        max_angle = self._config.max_gravity_angle_deg
        while self._verifications:
            plane = self._verifications.popleft()
            plane_eq = np.array([*plane.normal, plane.offset_mm])
            self._verification_output.send(
                SupportPlaneVerificationResult(
                    plane,
                    gravity_angle_deg(plane_eq, gravity),
                    is_valid_horizontal_plane(plane_eq, gravity, max_angle),
                )
            )

    def _process_cloud(self, cloud: dai.ADatatype, imu: dai.IMUData) -> None:
        gravity = _gravity(imu)
        if gravity is None:
            return
        if not isinstance(cloud, dai.PointCloudData):
            self._finish(
                CalibrationOutcome.FAILED,
                [],
                "Calibration failed: expected organized PointCloudData",
            )
            return
        assert self._fitter is not None
        try:
            candidates = self._fitter.find_candidates(
                np.asarray(cloud.getPoints()), gravity
            )
        except (ValueError, RuntimeError) as exc:
            self._finish(CalibrationOutcome.FAILED, [], f"Calibration failed: {exc}")
            return
        if candidates:
            self._finish(
                CalibrationOutcome.SUCCEEDED, candidates, "Support planes found"
            )


def _gravity(imu: dai.IMUData) -> Optional[np.ndarray]:
    """Raw accelerometer gravity from the first packet that carries it."""
    for packet in imu.packets:
        acceleration = packet.acceleroMeter
        if acceleration is not None:
            return np.array(
                [acceleration.x, acceleration.y, acceleration.z], dtype=np.float32
            )
    return None
