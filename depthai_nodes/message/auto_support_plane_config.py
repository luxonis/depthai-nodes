"""Build-time and runtime settings for ``AutoSupportPlaneCalibration``."""

from dataclasses import dataclass, field, replace
from enum import Enum

import depthai as dai

from .plane_calibration_config import (
    PlaneCalibrationConfig,
    SupportPlaneCalibrationConfig,
    _check_float,
    _check_int,
)

_ACCEPTANCE_FIELDS = (
    "vertical_distance_range_mm",
    "min_on_plane_ratio",
    "min_sampled_point_count",
    "check_distance_mm",
    "max_gravity_angle_deg",
)


@dataclass(frozen=True, kw_only=True)
class AutoSupportPlaneConfig(dai.Buffer):
    """Complete settings for ``AutoSupportPlaneCalibration``.

    Pass this message to ``build()`` or send it to ``inputConfig`` to replace
    the current settings. The type of ``calibration`` selects the backend at
    build and must stay the same for runtime updates.

    Attributes:
        mode: Acquire a plane once with ``Mode.ON_START``, or also check and
            refit it with ``Mode.CONTINUOUS``.
        check_interval_sec: Positive delay from a check result to the next check.
        failed_checks_before_refit: Consecutive failed checks that trigger a
            refit in continuous mode. At least 1.
        calibration: Inner calibration settings. A ``PlaneCalibrationConfig``
            selects the legacy calibrator.
        vertical_distance_range_mm: Inclusive range along gravity below the camera.
        min_on_plane_ratio: Minimum fraction of sampled points on the plane.
        min_sampled_point_count: Minimum number of valid sampled points.
        check_distance_mm: Maximum perpendicular distance for an on-plane point.
        max_gravity_angle_deg: Maximum plane tilt from horizontal.

    An omitted acceptance limit inherits the shared calibration setting. In
    legacy mode, supplying any acceptance limit enables the shared assessment
    in addition to the legacy fitter's requirements.
    """

    class Mode(str, Enum):
        """How the node maintains the support plane."""

        ON_START = "on_start"
        CONTINUOUS = "continuous"

    mode: Mode = Mode.ON_START
    check_interval_sec: float = 0.5
    failed_checks_before_refit: int = 3
    calibration: SupportPlaneCalibrationConfig | PlaneCalibrationConfig = field(
        default_factory=SupportPlaneCalibrationConfig
    )
    vertical_distance_range_mm: tuple[float, float] | None = None
    min_on_plane_ratio: float | None = None
    min_sampled_point_count: int | None = None
    check_distance_mm: float | None = None
    max_gravity_angle_deg: float | None = None
    _assessment: SupportPlaneCalibrationConfig | None = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        super().__init__()
        if not isinstance(self.mode, self.Mode):
            raise TypeError(
                f"mode must be AutoSupportPlaneConfig.Mode, got {type(self.mode).__name__}"
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

        overrides = {
            name: value
            for name in _ACCEPTANCE_FIELDS
            if (value := getattr(self, name)) is not None
        }
        calibration = self.calibration
        if isinstance(calibration, SupportPlaneCalibrationConfig):
            assessment = replace(calibration, **overrides)
        elif isinstance(calibration, PlaneCalibrationConfig):
            assessment = None
            if overrides:
                legacy_limits = SupportPlaneCalibrationConfig(
                    min_on_plane_ratio=calibration.min_on_plane_ratio,
                    check_distance_mm=calibration.check_distance_mm,
                    max_gravity_angle_deg=calibration.max_gravity_angle_deg,
                )
                assessment = replace(legacy_limits, **overrides)
        else:
            raise TypeError(
                "calibration must be SupportPlaneCalibrationConfig or PlaneCalibrationConfig, got "
                f"{type(calibration).__name__}"
            )
        object.__setattr__(self, "_assessment", assessment)
        if assessment is not None:
            for name in overrides:
                object.__setattr__(self, name, getattr(assessment, name))

    def _acceptance_limits(self) -> tuple:
        limits = tuple(
            getattr(self._assessment, name, None) for name in _ACCEPTANCE_FIELDS
        )
        calibration = self.calibration
        if isinstance(calibration, PlaneCalibrationConfig):
            return limits + (
                calibration.min_camera_height_mm,
                calibration.max_camera_height_mm,
                calibration.max_gravity_angle_deg,
                calibration.check_distance_mm,
                calibration.min_on_plane_ratio,
            )
        return limits


class _AutoConfigUpdate(dai.Buffer):
    """Apply a versioned settings snapshot on the calibrator's control queue."""

    def __init__(
        self, config: AutoSupportPlaneConfig, revision: int, searching: bool
    ) -> None:
        super().__init__()
        self.config = config
        self.revision = revision
        self.searching = searching
