"""Settings for ``AutoSupportPlane``."""

from dataclasses import dataclass, field
from enum import Enum

from .plane_calibration_config import (
    PlaneCalibrationConfig,
    _check_float,
    _check_int,
)


@dataclass(frozen=True, kw_only=True)
class AutoSupportPlaneConfig:
    """Settings for ``AutoSupportPlane``, fixed at build.

    Attributes:
        mode: `Mode.ON_START` finds a support plane once and holds it;
            `Mode.CONTINUOUS` also runs plane checks and refits after repeated
            failed checks.
        check_interval_sec: Seconds from one plane check's answer to the next
            plane check, in ``CONTINUOUS`` mode. Must be finite and positive.
        failed_checks_before_refit: Failed plane checks in a row that start a
            refit, in ``CONTINUOUS`` mode. At least 1.
        calibration: Settings of the inner ``SupportPlaneCalibrator``,
            including the ``check_distance_mm`` and ``min_on_plane_ratio`` that
            judge a plane check. The on-plane ratio is not a candidate's
            ``inlier_ratio``.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is not finite or out of range.
    """

    class Mode(str, Enum):
        """How ``AutoSupportPlane`` keeps its support plane."""

        ON_START = "on_start"
        CONTINUOUS = "continuous"

    mode: Mode = Mode.ON_START
    check_interval_sec: float = 0.5
    failed_checks_before_refit: int = 3
    calibration: PlaneCalibrationConfig = field(default_factory=PlaneCalibrationConfig)

    def __post_init__(self) -> None:
        """Validate the fields and store numbers as built-in types.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is not finite or out of range.
        """
        if not isinstance(self.mode, AutoSupportPlaneConfig.Mode):
            raise TypeError(
                "mode must be AutoSupportPlaneConfig.Mode, got "
                f"{type(self.mode).__name__}"
            )
        interval = _check_float("check_interval_sec", self.check_interval_sec)
        if interval <= 0:
            raise ValueError(f"check_interval_sec must be positive, got {interval}")
        object.__setattr__(self, "check_interval_sec", interval)
        checks = _check_int(
            "failed_checks_before_refit", self.failed_checks_before_refit
        )
        if checks < 1:
            raise ValueError(
                f"failed_checks_before_refit must be at least 1, got {checks}"
            )
        object.__setattr__(self, "failed_checks_before_refit", checks)
        if not isinstance(self.calibration, PlaneCalibrationConfig):
            raise TypeError(
                "calibration must be PlaneCalibrationConfig, got "
                f"{type(self.calibration).__name__}"
            )
