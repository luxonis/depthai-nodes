import time
from collections import deque
from enum import Enum
from typing import cast

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
from .support_plane_fitter import SupportPlaneFitter, _valid_points

# Oldest IMU sample, by host arrival time, used for gravity. The IMU streams at
# tens to hundreds of hertz, so an older sample means the stream has stalled.
IMU_MAX_AGE_SECONDS = 0.5

# Shortest accelerometer vector, in m/s², read as gravity rather than as a
# packet without an accelerometer report.
MIN_GRAVITY_NORM = 1e-6

# The on-plane ratio samples every this-many valid points of the flat point
# array, from the first one.
ON_PLANE_STRIDE = 16
# The on-plane ratio is not measured under this many sampled points, about
# 8000 valid points.
MIN_ON_PLANE_POINTS = 500


class CalibratorState(Enum):
    """Where the calibrator is in an attempt; see `SupportPlaneCalibrator`."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"


class SupportPlaneCalibrator(BaseThreadedHostNode):
    """Find the support plane in shared coordinates, checked against gravity.

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

    Errors: these raise from the running node.
        - A message on the point cloud input that is not
          ``dai.PointCloudData``: ``TypeError``.
        - The IMU's device has no IMU extrinsics: ``RuntimeError`` naming the
          device ID.
        - A multi-device calibration is set without the IMU's device:
          ``RuntimeError`` naming the device ID.
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
        self._imu_device: dai.Device | None = None
        # IMU axes to shared coordinates, set when the node starts running.
        self._imu_rotation: np.ndarray | None = None
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
        config: PlaneCalibrationConfig | None = None,
    ) -> "SupportPlaneCalibrator":
        """Link the inputs and fix the fitting settings.

        Args:
            pointCloud: Output emitting ``dai.PointCloudData`` in millimeters
                and in shared coordinates, organized or not. A point counts
                when it is finite and not ``(0, 0, 0)``. It must share one frame
                with the boxes the plane serves (same depth outputs, same
                multi-device calibration); this is not checked. Without a
                multi-device calibration, shared coordinates are the frame of
                the IMU device's calibration-origin camera, CAM_A on most
                devices.
            imu: Output emitting ``dai.IMUData`` with ``ACCELEROMETER_RAW``
                packets from one device's IMU; for a host source, the
                pipeline's default device. Gravity is taken in shared
                coordinates, so set any multi-device calibration before the
                pipeline starts.
            config: Plane-fitting and attempt settings; ``None`` uses the
                defaults.

        Returns:
            The configured node instance.

        Raises:
            ImportError: If open3d is not installed.
            RuntimeError: If the node is already built.
            TypeError: If ``config`` is not a ``PlaneCalibrationConfig``.
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
        if config is None:
            config = PlaneCalibrationConfig()
        if not isinstance(config, PlaneCalibrationConfig):
            raise TypeError(
                "config must be PlaneCalibrationConfig, got " f"{type(config).__name__}"
            )
        self._config = config
        self._fitter = SupportPlaneFitter(config)
        source = imu.getParent()
        if source.runOnHost():
            self._imu_device = self.getParentPipeline().getDefaultDevice()
        else:
            # A node that doesn't run on the host is a device node.
            self._imu_device = cast(dai.DeviceNode, source).getDevice()
        pointCloud.link(self._point_cloud_input)
        imu.link(self._imu_input)
        return self

    def run(self) -> None:
        """Apply controls, answer verifications and fit clouds until stopped."""
        assert self._imu_device is not None
        self._imu_rotation = _imu_to_shared_rotation(
            self._imu_device, self.getParentPipeline().getMultiDeviceCalibration()
        )
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
        cloud, points = _cloud_points(message)
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
        assert self._imu_rotation is not None, "run() sets the IMU rotation"
        now = time.monotonic()
        while True:
            imu = self._imu_input.tryGet()
            if imu is None:
                break
            if not isinstance(imu, dai.IMUData):
                continue
            gravity = imu_gravity(imu)
            if gravity is not None:
                self._latest_gravity = self._imu_rotation @ gravity
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

    def _answer_verifications(
        self, cloud: dai.PointCloudData, points: np.ndarray, gravity: np.ndarray
    ) -> None:
        """Answer all pending verifications from one cloud, in arrival order."""
        assert self._config is not None and self._fitter is not None
        if not self._verifications:
            return
        sample = _sample_valid_points(points)
        while self._verifications:
            plane = self._verifications.popleft()
            plane_eq = orient_along_gravity(
                np.array([*plane.normal, plane.offset_mm]), gravity
            )
            on_plane_ratio = _on_plane_ratio(
                plane, sample, self._config.check_distance_mm
            )
            accepted = (
                is_valid_horizontal_plane(plane_eq, gravity, self._config)
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


def _imu_to_shared_rotation(
    device: dai.Device, multi_device_calibration: list[dai.MultiDeviceExtrinsics] | None
) -> np.ndarray:
    """Return the rotation taking ``device``'s IMU axes to shared coordinates.

    Raises:
        RuntimeError: If ``device`` has no IMU extrinsics, or
            ``multi_device_calibration`` is set without that device.
    """
    device_id = device.getDeviceId()
    calibration = device.getCalibration()
    socket: dai.CameraBoardSocket | None = None
    camera_to_shared = np.eye(3, dtype=np.float32)
    if multi_device_calibration is not None:
        handler = dai.beta.MultiDeviceCalibrationHandler(multi_device_calibration)
        device_socket = handler.getDeviceSocket(device_id)
        if device_socket is None:
            raise RuntimeError(
                f"Device {device_id} supplies the IMU but is not in the "
                "pipeline's multi-device calibration"
            )
        socket = device_socket
        to_origin = handler.getExtrinsicsToOrigin(device_id, socket)
        # DepthAI returns extrinsics for every device that has a socket.
        assert to_origin is not None
        camera_to_shared = np.asarray(to_origin.getRotationMatrix(), dtype=np.float32)
    try:
        if socket is None:
            # With one device, shared coordinates are its local calibration origin.
            # DepthAI's stub types the (matrix, origin socket) result as a bare tuple.
            _, origin = calibration.getExtrinsicsToOrigin(
                calibration.getCameraWithLowestId()
            )
            socket = cast(dai.CameraBoardSocket, origin)
        imu_to_camera = calibration.getImuToCameraExtrinsics(socket)
    except RuntimeError as e:
        raise RuntimeError(
            f"The calibration of device {device_id} has no IMU extrinsics, so its "
            "gravity cannot be put in shared coordinates"
        ) from e
    return camera_to_shared @ np.asarray(imu_to_camera, dtype=np.float32)[:3, :3]


def _cloud_points(message: dai.ADatatype) -> tuple[dai.PointCloudData, np.ndarray]:
    """Return a point cloud message and its points.

    Raises:
        TypeError: If ``message`` is not ``dai.PointCloudData``.
    """
    if not isinstance(message, dai.PointCloudData):
        raise TypeError(
            "SupportPlaneCalibrator expects PointCloudData on its "
            f"point cloud input, got {type(message).__name__}"
        )
    return message, np.asarray(message.getPoints())


def _sample_valid_points(points: np.ndarray) -> np.ndarray:
    """Return every ``ON_PLANE_STRIDE``-th valid point, from the first one."""
    return _valid_points(points)[::ON_PLANE_STRIDE].astype(np.float64)


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
