"""Host messages for camera installation monitoring."""

from dataclasses import dataclass
from enum import Enum

import depthai as dai


class CameraMovementControl(dai.Buffer):
    """Request calibration using IMU reports captured from this timestamp onward."""

    @classmethod
    def calibrate(cls) -> "CameraMovementControl":
        """Create a calibration request stamped in the DepthAI host clock."""
        message = cls()
        message.setTimestamp(dai.Clock.now())
        return message


class CameraMovementReason(str, Enum):
    """Evidence that triggered the movement alert."""

    TILT_CHANGED = "tilt_changed"
    TURN_DETECTED = "turn_detected"


class CameraMovementCalibrationState(str, Enum):
    """State of the latest calibration request."""

    NOT_REQUESTED = "not_requested"
    COLLECTING = "collecting"
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"


@dataclass(eq=False, kw_only=True)
class CameraMovementEvent(dai.Buffer):
    """A tilt alert or a completed turn, with its estimated angle in degrees."""

    reason: CameraMovementReason
    angle_deg: float

    def __post_init__(self) -> None:
        super().__init__()


@dataclass(eq=False, kw_only=True)
class CameraMovementStatus(dai.Buffer):
    """A heartbeat with the last estimate, calibration state and IMU freshness.

    Sample ages measure host time since the last usable report arrived. A stale stream
    can retain a baseline and a latched tilt alarm.
    """

    calibration: CameraMovementCalibrationState
    baseline_ready: bool
    tilt_changed: bool
    tilt_deg: float | None
    accelerometer_age_s: float | None
    gyroscope_age_s: float | None
    imu_stale: bool

    def __post_init__(self) -> None:
        super().__init__()
