"""Monitor a fixed camera against an explicitly calibrated orientation."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import timedelta

import depthai as dai
import numpy as np

from depthai_nodes.message.camera_movement import (
    CameraMovementCalibrationState,
    CameraMovementControl,
    CameraMovementEvent,
    CameraMovementReason,
    CameraMovementStatus,
)

_GRAVITY = 9.80665
_GRAVITY_TOLERANCE = 0.6  # m/s²
_STATIONARY_RATE = math.radians(3)
_MAX_TILT_RATE = math.radians(8)
_CALIBRATION_SPREAD_DEG = 1.0
_GRAVITY_FILTER_SECONDS = 0.2
_TURN_SETTLE_SECONDS = 0.5
_MAX_TURN_SECONDS = 15.0
_ZERO_TIME = timedelta(0)


@dataclass(frozen=True, kw_only=True)
class CameraMovementConfig:
    """Thresholds and timing for fixed-camera monitoring.

    The tilt alert rearms below `tilt_rearm_deg`. Set
    `turn_threshold_deg` to detect net rotation around gravity, or leave it
    at None for tilt monitoring alone. A turn starts at a projected rate of
    at least 3°/s and must finish with 0.5 seconds below that rate within 15
    seconds of starting. Setting `status_interval_s` to None disables the
    heartbeat. A gap longer than `max_sample_gap_s` in either IMU stream
    interrupts calibration, confirmation and turns.
    """

    tilt_threshold_deg: float = 3.0
    tilt_rearm_deg: float = 1.5
    tilt_confirmation_s: float = 0.5
    turn_threshold_deg: float | None = None
    calibration_window_s: float = 1.0
    calibration_timeout_s: float = 8.0
    status_interval_s: float | None = 1.0
    max_sample_gap_s: float = 0.1
    imu_stale_after_s: float = 0.5

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value is None and name in ("turn_threshold_deg", "status_interval_s"):
                continue
            if (
                value is None
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if not self.tilt_rearm_deg < self.tilt_threshold_deg < 180:
            raise ValueError(
                "Tilt rearm must be below the alert threshold, which must be below 180°"
            )
        if self.calibration_window_s >= self.calibration_timeout_s:
            raise ValueError("Calibration timeout must exceed the stationary window")


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def _vector(
    report: dai.IMUReportAccelerometer | dai.IMUReportGyroscope,
) -> np.ndarray | None:
    xyz = np.array([report.x, report.y, report.z])
    if report.getTimestampDevice() > _ZERO_TIME and np.isfinite(xyz).all():
        return xyz
    return None


class CameraMovementDetector(dai.node.ThreadedHostNode):
    """Report sustained camera tilt from a fixed, app-calibrated baseline.

    Feed `dai.IMUData` containing accelerometer and gyro reports in the same
    coordinate frame. Send `CameraMovementControl.calibrate()` to
    `inputControl` at the intended pose. Successful calibration replaces the
    reference and rearms the alert. A timeout leaves any prior reference intact.

    `out` emits `CameraMovementEvent`. `status` sends
    `CameraMovementStatus` even while IMU input is stalled. Optional gyro
    turns measure short episodes around gravity and do not track absolute yaw.
    """

    def __init__(self) -> None:
        super().__init__()
        self._imu = self.createInput(name="imu", queueSize=128, blocking=False)
        self.inputControl = self.createInput(name="control")
        self.out = self.createOutput(name="out")
        self.status = self.createOutput(name="status")
        self._config = CameraMovementConfig()
        self._baseline: np.ndarray | None = None
        self._filtered_accel: np.ndarray | None = None
        self._gyro_bias = np.zeros(3)
        self._tilt_deg: float | None = None
        self._tilt_changed = False
        self._tilt_candidate: float | None = None
        self._calibration = CameraMovementCalibrationState.NOT_REQUESTED
        self._calibration_after = _ZERO_TIME
        self._calibration_deadline = 0.0
        self._calibration_start: float | None = None
        self._calibration_first_gravity = np.zeros(3)
        self._calibration_gravity_sum = np.zeros(3)
        self._calibration_gyro_sum = np.zeros(3)
        self._calibration_count = 0
        self._gyro: np.ndarray | None = None
        self._gyro_time = 0.0
        self._gyro_host_time = _ZERO_TIME
        self._accel_time = 0.0
        self._gyro_arrival: float | None = None
        self._accel_arrival: float | None = None
        self._pending_accel: deque[
            tuple[dai.IMUReportAccelerometer, np.ndarray, float, float]
        ] = deque()
        self._turn_gravity: np.ndarray | None = None
        self._turn_start: float | None = None
        self._turn_quiet_since: float | None = None
        self._turn_angle = 0.0

    def build(
        self, imu: dai.Node.Output, config: CameraMovementConfig | None = None
    ) -> CameraMovementDetector:
        """Link an IMU stream and optionally configure the detector."""
        if config is not None:
            self._config = config
        imu.link(self._imu)
        return self

    def run(self) -> None:
        """Service controls and IMU batches and publish independent heartbeats."""
        next_status = time.monotonic()
        try:
            while self.isRunning():
                now = time.monotonic()
                while (command := self.inputControl.tryGet()) is not None:
                    if isinstance(command, CameraMovementControl):
                        self._calibration = CameraMovementCalibrationState.COLLECTING
                        self._calibration_after = (
                            command.getTimestamp() or dai.Clock.now()
                        )
                        self._calibration_deadline = (
                            now + self._config.calibration_timeout_s
                        )
                        self._calibration_start = None
                if (
                    self._calibration is CameraMovementCalibrationState.COLLECTING
                    and now >= self._calibration_deadline
                ):
                    self._calibration = CameraMovementCalibrationState.TIMED_OUT
                message = self._imu.tryGet()
                if isinstance(message, dai.IMUData):
                    for packet in message.packets:
                        self._process_packet(packet, now)
                interval = self._config.status_interval_s
                if interval is not None and now >= next_status:
                    self._send_status(now)
                    next_status = now + interval
                if message is None:
                    time.sleep(0.005)
        except dai.MessageQueue.QueueException:  # pyright: ignore[reportGeneralTypeIssues]
            return

    def _process_packet(self, packet: dai.IMUPacket, now: float) -> None:
        self._queue_accel(packet.acceleroMeter, now)
        self._process_gyro(packet.gyroscope, now)
        self._process_ready_accel()

    def _process_gyro(self, report: dai.IMUReportGyroscope, now: float) -> None:
        sample = _vector(report)
        stamp = report.getTimestampDevice().total_seconds()
        if sample is None or stamp <= self._gyro_time:
            return
        dt = stamp - self._gyro_time
        self._gyro = sample
        self._gyro_time = stamp
        self._gyro_host_time = report.getTimestamp()
        self._gyro_arrival = now
        if dt > self._config.max_sample_gap_s:
            self._reset_imu_windows()
            while self._pending_accel and self._pending_accel[0][2] < stamp:
                self._pending_accel.popleft()
            return

        threshold = self._config.turn_threshold_deg
        gravity = self._turn_gravity
        if threshold is None or gravity is None:
            return
        if abs(stamp - self._accel_time) > self._config.max_sample_gap_s:
            self._turn_gravity = None
            self._turn_start = None
            return

        # Gravity rotates opposite to the camera in IMU coordinates. Propagate
        # it with the gyro while the tilt filter is paused during fast motion.
        omega = sample - self._gyro_bias
        speed = float(np.linalg.norm(omega))
        rate = float(np.dot(omega, gravity))
        if speed > 0:
            axis = omega / speed
            cosine = math.cos(speed * dt)
            self._turn_gravity = (
                cosine * gravity
                - math.sin(speed * dt) * np.cross(axis, gravity)
                + (1 - cosine) * (rate / speed) * axis
            )
        if self._turn_start is None:
            if abs(rate) >= _STATIONARY_RATE:
                self._turn_start = stamp
                self._turn_quiet_since = None
                self._turn_angle = rate * dt
            return
        within_window = stamp - self._turn_start <= _MAX_TURN_SECONDS
        if within_window:
            self._turn_angle += rate * dt
        if abs(rate) >= _STATIONARY_RATE:
            self._turn_quiet_since = None
        elif self._turn_quiet_since is None:
            self._turn_quiet_since = stamp
        elif stamp - self._turn_quiet_since >= _TURN_SETTLE_SECONDS:
            angle = math.degrees(abs(self._turn_angle))
            if within_window and angle >= threshold:
                self._send_event(CameraMovementReason.TURN_DETECTED, angle, report)
            self._turn_start = None

    def _reset_imu_windows(self) -> None:
        self._calibration_start = None
        self._tilt_candidate = None
        self._turn_start = None
        self._turn_gravity = None

    def _queue_accel(self, report: dai.IMUReportAccelerometer, now: float) -> None:
        sample = _vector(report)
        stamp = report.getTimestampDevice().total_seconds()
        if sample is None or stamp <= self._accel_time:
            return
        dt = stamp - self._accel_time
        self._accel_time = stamp
        self._accel_arrival = now
        gap = self._config.max_sample_gap_s
        if dt > gap:
            self._reset_imu_windows()
            self._pending_accel.clear()
        while self._pending_accel and stamp - self._pending_accel[0][2] > gap:
            self._pending_accel.popleft()
        self._pending_accel.append((report, sample, stamp, dt))

    def _process_ready_accel(self) -> None:
        # Wait for gyro coverage before using an accelerometer report to
        # confirm a tilt or complete calibration. The next gyro report can
        # reveal a gap even when the cached report is still close in time.
        while self._pending_accel and self._pending_accel[0][2] <= self._gyro_time:
            self._process_accel(*self._pending_accel.popleft())

    def _process_accel(
        self,
        report: dai.IMUReportAccelerometer,
        sample: np.ndarray,
        stamp: float,
        dt: float,
    ) -> None:
        magnitude = float(np.linalg.norm(sample))
        gyro = self._gyro
        if (
            gyro is None
            or abs(stamp - self._gyro_time) > self._config.max_sample_gap_s
            or abs(magnitude - _GRAVITY) > _GRAVITY_TOLERANCE
        ):
            self._calibration_start = None
            self._tilt_candidate = None
            return

        direction = sample / magnitude
        if self._calibration is CameraMovementCalibrationState.COLLECTING:
            self._collect_calibration(report, stamp, direction, gyro)
        if self._baseline is None or self._filtered_accel is None:
            return
        speed = float(np.linalg.norm(gyro - self._gyro_bias))
        if speed > _MAX_TILT_RATE:
            self._tilt_candidate = None
            return

        weight = -math.expm1(-dt / _GRAVITY_FILTER_SECONDS)
        self._filtered_accel += weight * (direction - self._filtered_accel)
        magnitude = np.linalg.norm(self._filtered_accel)
        if magnitude < 0.5:
            self._tilt_candidate = None
            return
        if (
            self._config.turn_threshold_deg is not None
            and self._turn_start is None
            and speed < _STATIONARY_RATE
        ):
            if self._turn_gravity is None:
                self._turn_gravity = direction
            else:
                gravity = self._turn_gravity + weight * (direction - self._turn_gravity)
                self._turn_gravity = gravity / np.linalg.norm(gravity)
        self._tilt_deg = _angle_deg(self._baseline, self._filtered_accel / magnitude)
        if self._tilt_changed:
            beyond = self._tilt_deg <= self._config.tilt_rearm_deg
        else:
            beyond = self._tilt_deg >= self._config.tilt_threshold_deg
        if not beyond:
            self._tilt_candidate = None
        elif self._tilt_candidate is None:
            self._tilt_candidate = stamp
        elif stamp - self._tilt_candidate >= self._config.tilt_confirmation_s:
            self._tilt_candidate = None
            self._tilt_changed = not self._tilt_changed
            if self._tilt_changed:
                self._send_event(
                    CameraMovementReason.TILT_CHANGED, self._tilt_deg, report
                )

    def _collect_calibration(
        self,
        report: dai.IMUReportAccelerometer,
        stamp: float,
        direction: np.ndarray,
        gyro: np.ndarray,
    ) -> None:
        if (
            min(report.getTimestamp(), self._gyro_host_time) < self._calibration_after
            or np.linalg.norm(gyro) >= _STATIONARY_RATE
        ):
            self._calibration_start = None
            return
        if (
            self._calibration_start is None
            or _angle_deg(self._calibration_first_gravity, direction)
            > _CALIBRATION_SPREAD_DEG
        ):
            self._calibration_start = stamp
            self._calibration_first_gravity = direction
            self._calibration_gravity_sum = direction.copy()
            self._calibration_gyro_sum = gyro.copy()
            self._calibration_count = 1
            return
        self._calibration_gravity_sum += direction
        self._calibration_gyro_sum += gyro
        self._calibration_count += 1
        if stamp - self._calibration_start >= self._config.calibration_window_s:
            baseline = self._calibration_gravity_sum
            self._baseline = baseline / np.linalg.norm(baseline)
            self._turn_gravity = self._baseline
            self._filtered_accel = self._baseline.copy()
            self._gyro_bias = self._calibration_gyro_sum / self._calibration_count
            self._calibration = CameraMovementCalibrationState.SUCCEEDED
            self._tilt_deg = 0.0
            self._tilt_changed = False
            self._tilt_candidate = None
            self._turn_start = None

    def _send_event(
        self, reason: CameraMovementReason, angle: float, report: dai.IMUReport
    ) -> None:
        event = CameraMovementEvent(reason=reason, angle_deg=angle)
        event.setTimestamp(report.getTimestamp())
        event.setTimestampDevice(report.getTimestampDevice())
        event.setSequenceNum(report.getSequenceNum())
        self.out.send(event)

    def _send_status(self, now: float) -> None:
        accel_age = None if self._accel_arrival is None else now - self._accel_arrival
        gyro_age = None if self._gyro_arrival is None else now - self._gyro_arrival
        status = CameraMovementStatus(
            calibration=self._calibration,
            baseline_ready=self._baseline is not None,
            tilt_changed=self._tilt_changed,
            tilt_deg=self._tilt_deg,
            accelerometer_age_s=accel_age,
            gyroscope_age_s=gyro_age,
            imu_stale=any(
                age is None or age >= self._config.imu_stale_after_s
                for age in (accel_age, gyro_age)
            ),
        )
        status.setTimestamp(dai.Clock.now())
        self.status.send(status)
