import numbers
import time
from collections import deque
from collections.abc import Sequence
from enum import Enum
from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.support_plane import SupportPlane, SupportPlaneCandidate
from depthai_nodes.message.support_plane_calibration import (
    CalibrationOutcome,
    SupportPlaneCalibrationControl,
    SupportPlaneCalibrationResult,
    SupportPlaneVerificationResult,
)
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode

from .plane_calibration_config import PlaneCalibrationConfig
from .plane_validation import (
    gravity_angle_deg,
    is_valid_horizontal_plane,
    orient_along_gravity,
)
from .support_plane_fitter import SupportPlaneFitter


def _check_size(name: str, value: Any) -> tuple[int, int]:
    """Return a ``(width, height)`` sequence of positive integers as a tuple.

    Raises:
        TypeError: If the value is not a sequence, is a ``str`` or ``bytes``, or
            holds elements that are ``bool`` or not integers.
        ValueError: If the value does not have exactly two positive elements.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a (width, height) sequence, got {type(value).__name__}"
        )
    if len(value) != 2:
        raise ValueError(f"{name} must be (width, height), got {len(value)} values")
    for label, element in zip(("width", "height"), value):
        if isinstance(element, bool) or not isinstance(element, numbers.Integral):
            raise TypeError(
                f"{name} {label} must be an integer, got {type(element).__name__}"
            )
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive, got ({width}, {height})")
    return width, height


# Oldest IMU sample, by host arrival time, used for gravity. The IMU streams at
# tens to hundreds of hertz, so an older sample means the stream has stalled.
IMU_MAX_AGE_SECONDS = 0.5

# Shortest accelerometer vector, in m/s², read as gravity rather than as a
# packet without an accelerometer report.
MIN_GRAVITY_NORM = 1e-6

# The on-plane ratio samples every this-many rows and columns of the organized
# cloud, and is not measured under this many valid sampled points.
ON_PLANE_STRIDE = 4
MIN_ON_PLANE_POINTS = 500


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
    ``pip install "depthai-nodes[box-measurement]"``.

    States:
        - Idle: Start begins an attempt.
        - Running: fits point clouds until one yields candidates (succeeded),
          Stop arrives (cancelled), or the attempt cap is reached (failed).
          Stop takes effect after the current fit.
        - Complete: after success or failure; Reset returns to idle. A
          cancelled attempt returns straight to idle.

    Verify commands are answered in any state, on the next point cloud taken
    with fresh IMU, before that cloud is fitted. Fitting and verification wait
    for an IMU sample younger than ``IMU_MAX_AGE_SECONDS`` with a nonzero
    accelerometer vector.

    Errors: a message on the point cloud input that is not
    ``dai.PointCloudData`` raises ``TypeError``, and a cloud that is not the
    organized size from `build` raises ``ValueError``. An exception escaping
    ``run()`` aborts the process on depthai 3.12.
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
        self._config: PlaneCalibrationConfig | None = None
        self._fitter: SupportPlaneFitter | None = None
        self._verifications: deque[SupportPlane] = deque()
        self._latest_gravity: np.ndarray | None = None
        self._latest_gravity_arrival = 0.0
        self._last_fitted_cloud: dai.PointCloudData | None = None
        # Clouds the running attempt has fitted, checked against the cap.
        self._fitted_cloud_count = 0

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
        pointCloudSize: tuple[int, int],
        config: PlaneCalibrationConfig | None = None,
    ) -> "SupportPlaneCalibrator":
        """Link the inputs and fix the cloud size and fitting settings.

        Args:
            pointCloud: Output emitting organized ``dai.PointCloudData`` in
                millimeters.
            imu: Output emitting ``dai.IMUData`` with accelerometer packets.
            pointCloudSize: Cloud ``(width, height)`` in points.
            config: Plane-fitting and attempt settings; ``None`` uses the
                defaults.

        Returns:
            The configured node instance.

        Raises:
            ImportError: If open3d is not installed.
            RuntimeError: If the node is already built.
            TypeError: If ``pointCloudSize`` is not a pair of integers or
                ``config`` is not a ``PlaneCalibrationConfig``.
            ValueError: If ``pointCloudSize`` does not have two positive values.
        """
        try:
            import open3d  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__} requires open3d. Install it with "
                '`pip install "depthai-nodes[box-measurement]"`.'
            ) from e
        if self._fitter is not None:
            raise RuntimeError(
                "SupportPlaneCalibrator is already built; create a new node to "
                "change its configuration"
            )
        width, height = _check_size("pointCloudSize", pointCloudSize)
        if config is None:
            config = PlaneCalibrationConfig()
        if not isinstance(config, PlaneCalibrationConfig):
            raise TypeError(
                "config must be PlaneCalibrationConfig, got " f"{type(config).__name__}"
            )
        self._config = config
        self._fitter = SupportPlaneFitter((width, height), config)
        pointCloud.link(self._point_cloud_input)
        imu.link(self._imu_input)
        return self

    def run(self) -> None:
        """Apply controls, answer verifications and fit clouds until stopped."""
        try:
            while self.isRunning():
                self._run_pass()
                # Pause between passes, as DepthAI's DynamicCalibration does.
                time.sleep(0.005)
        # DepthAI's stub declares QueueException through an undeclared nested
        # class, so pyright cannot tell it is an exception; at runtime it is one.
        except dai.MessageQueue.QueueException:  # pyright: ignore[reportGeneralTypeIssues]
            return

    def _run_pass(self) -> None:
        """Apply controls, then check one cloud for verifications and the fit.

        A cloud is taken only with fresh gravity, and only while a verification is
        pending or an attempt is running. Otherwise it stays queued.
        """
        self._apply_controls()
        self._take_latest_gravity()
        gravity = self._latest_gravity
        if gravity is None:
            return
        if not self._verifications and self._state is not CalibratorState.RUNNING:
            return
        message = self._point_cloud_input.tryGet()
        if message is None:
            return
        cloud, points = self._check_cloud(message)
        self._answer_verifications(cloud, points, gravity)
        if self._state is CalibratorState.RUNNING:
            self._fit(cloud, points, gravity)

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
                    self._last_fitted_cloud = None
                    self._fitted_cloud_count = 0
            elif isinstance(command, commands.Stop):
                if self._state is CalibratorState.RUNNING:
                    self._finish(CalibrationOutcome.CANCELLED, [])
            elif self._state is not CalibratorState.RUNNING:
                self._state = CalibratorState.IDLE

    def _take_latest_gravity(self) -> None:
        """Keep the newest IMU gravity and drop it once it is too old."""
        now = time.monotonic()
        while True:
            imu = self._imu_input.tryGet()
            if imu is None:
                break
            if not isinstance(imu, dai.IMUData):
                continue
            gravity = imu_gravity(imu)
            if gravity is not None:
                self._latest_gravity = gravity
                self._latest_gravity_arrival = now
        if (
            self._latest_gravity is not None
            and now - self._latest_gravity_arrival > IMU_MAX_AGE_SECONDS
        ):
            self._latest_gravity = None

    def _finish(
        self,
        outcome: CalibrationOutcome,
        candidates: list[SupportPlaneCandidate],
    ) -> None:
        """End the attempt and send its result."""
        # Like DepthAI MultiDeviceCalibration, a stopped attempt needs no reset.
        self._state = (
            CalibratorState.IDLE
            if outcome is CalibrationOutcome.CANCELLED
            else CalibratorState.COMPLETE
        )
        result = SupportPlaneCalibrationResult(outcome, candidates)
        cloud = self._last_fitted_cloud
        if cloud is not None:
            result.setTimestamp(cloud.getTimestamp())
            result.setTimestampDevice(cloud.getTimestampDevice())
            result.setSequenceNum(cloud.getSequenceNum())
        self._out.send(result)

    def _check_cloud(
        self, message: dai.ADatatype
    ) -> tuple[dai.PointCloudData, np.ndarray]:
        """Return a point cloud message and its points, checking both.

        Raises:
            TypeError: If ``message`` is not ``dai.PointCloudData``.
            ValueError: If the cloud is not the organized size from `build`.
        """
        if not isinstance(message, dai.PointCloudData):
            raise TypeError(
                "SupportPlaneCalibrator expects organized PointCloudData on its "
                f"point cloud input, got {type(message).__name__}"
            )
        assert self._fitter is not None
        width, height = self._fitter.cloud_size
        points = np.asarray(message.getPoints())
        expected = (width * height, 3)
        if points.shape != expected:
            raise ValueError(
                f"Point cloud is not organized to {(width, height)}: "
                f"got shape {points.shape}, expected {expected}"
            )
        return message, points

    def _answer_verifications(
        self, cloud: dai.PointCloudData, points: np.ndarray, gravity: np.ndarray
    ) -> None:
        """Answer all pending verifications from one cloud, in arrival order."""
        assert self._config is not None and self._fitter is not None
        if not self._verifications:
            return
        max_angle = self._config.max_gravity_angle_deg
        sample = _sample_valid_points(points, self._fitter.cloud_size)
        while self._verifications:
            plane = self._verifications.popleft()
            plane_eq = orient_along_gravity(
                np.array([*plane.normal, plane.offset_mm]), gravity
            )
            on_plane_ratio = _on_plane_ratio(
                plane, sample, self._config.check_distance_mm
            )
            accepted = (
                is_valid_horizontal_plane(plane_eq, gravity, max_angle)
                and on_plane_ratio is not None
                and on_plane_ratio >= self._config.min_on_plane_ratio
            )
            result = SupportPlaneVerificationResult(
                plane,
                gravity_angle_deg(plane_eq, gravity),
                accepted,
                on_plane_ratio,
            )
            result.setTimestamp(cloud.getTimestamp())
            result.setTimestampDevice(cloud.getTimestampDevice())
            result.setSequenceNum(cloud.getSequenceNum())
            self._verification_output.send(result)

    def _fit(
        self, cloud: dai.PointCloudData, points: np.ndarray, gravity: np.ndarray
    ) -> None:
        """Fit one cloud and finish the attempt on candidates or at the cap."""
        assert self._fitter is not None and self._config is not None
        candidates = self._fitter.find_candidates(points, gravity)
        self._last_fitted_cloud = cloud
        self._fitted_cloud_count += 1
        cap = self._config.max_calibration_attempts
        if candidates:
            self._finish(CalibrationOutcome.SUCCEEDED, candidates)
        elif cap is not None and self._fitted_cloud_count >= cap:
            self._finish(CalibrationOutcome.FAILED, [])


def _sample_valid_points(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Return every ``ON_PLANE_STRIDE``-th row and column's points with ``z > 0``."""
    width, height = size
    grid = points.reshape(height, width, 3)[::ON_PLANE_STRIDE, ::ON_PLANE_STRIDE]
    sample = grid.reshape(-1, 3).astype(np.float64)
    return sample[sample[:, 2] > 0]


def _on_plane_ratio(
    plane: SupportPlane, sample: np.ndarray, check_distance_mm: float
) -> float | None:
    """Return the share of sampled points within the distance of the plane.

    Returns ``None`` under ``MIN_ON_PLANE_POINTS`` sampled points. The sign of
    the plane equation does not matter.
    """
    if len(sample) < MIN_ON_PLANE_POINTS:
        return None
    distances = np.abs(sample @ np.asarray(plane.normal) + plane.offset_mm)
    return float(np.count_nonzero(distances <= check_distance_mm) / len(sample))


def imu_gravity(imu: dai.IMUData) -> np.ndarray | None:
    """Return the raw accelerometer gravity from the first packet carrying it.

    A packet without an accelerometer report, for example from an IMU set up
    for the gyroscope only, has a zero accelerometer vector, so such vectors
    count as missing.

    Args:
        imu: IMU message to read.

    Returns:
        The accelerometer ``(x, y, z)`` vector, or ``None`` if no packet has a
        nonzero one.
    """
    for packet in imu.packets:
        acceleration = packet.acceleroMeter
        gravity = np.array(
            [acceleration.x, acceleration.y, acceleration.z], dtype=np.float32
        )
        if np.linalg.norm(gravity) >= MIN_GRAVITY_NORM:
            return gravity
    return None
