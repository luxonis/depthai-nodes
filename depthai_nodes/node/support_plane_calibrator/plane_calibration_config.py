"""Plane-fitting and attempt settings for ``SupportPlaneCalibrator``."""

import math
import numbers
from dataclasses import dataclass
from typing import Any


def _check_float(name: str, value: Any) -> float:
    """Return a finite real number, ``bool`` excluded, as a built-in ``float``.

    Raises:
        TypeError: If the value is a ``bool`` or not a ``numbers.Real``.
        ValueError: If the value is not finite.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def _check_int(name: str, value: Any) -> int:
    """Return an integral number, ``bool`` excluded, as a built-in ``int``.

    Raises:
        TypeError: If the value is a ``bool`` or not a ``numbers.Integral``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


@dataclass(frozen=True, kw_only=True)
class PlaneCalibrationConfig:
    """Plane-fitting and attempt settings for ``SupportPlaneCalibrator``.

    The settings are fixed at build.

    Attributes:
        distance_threshold_mm: Largest distance of a RANSAC inlier from the
            plane, in millimeters. Must be positive.
        ransac_sample_points: Points sampled per RANSAC iteration. At least 3.
        ransac_max_iterations: RANSAC iterations per plane. At least 1.
        max_plane_passes: Planes extracted from one point cloud. At least 1.
        min_inlier_ratio: Smallest share of the cloud a plane must hold, from 0
            to 1.
        voxel_size_mm: Downsampling voxel size in millimeters. Must be positive.
        max_gravity_angle_deg: Largest angle between a plane normal and the
            direction opposing gravity, from 0 up to but excluding 90.
        merge_distance_threshold_mm: Largest offset difference, in millimeters,
            at which two parallel candidates count as one plane. Must be
            positive.
        max_calibration_attempts: Point clouds fitted in one calibration
            attempt, not Starts, before an attempt without candidates ends
            ``FAILED``. At least 1, or ``None`` to search until stopped.
        check_distance_mm: Largest distance, in millimeters, of a point from a
            verified plane for it to count as on the plane in a plane
            verification's on-plane ratio. Separate from RANSAC's
            ``distance_threshold_mm``. Must be positive.
        min_on_plane_ratio: Smallest on-plane ratio, from 0 to 1, at which a
            plane verification accepts a plane that passes the level test.
            Separate from ``min_inlier_ratio``, which applies to fitting.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is not finite or out of range.
    """

    distance_threshold_mm: float = 10.0
    ransac_sample_points: int = 3
    ransac_max_iterations: int = 300
    max_plane_passes: int = 10
    min_inlier_ratio: float = 0.05
    voxel_size_mm: float = 30.0
    max_gravity_angle_deg: float = 6.5
    merge_distance_threshold_mm: float = 25.0
    max_calibration_attempts: int | None = None
    check_distance_mm: float = 30.0
    min_on_plane_ratio: float = 0.1

    def __post_init__(self) -> None:
        """Validate the fields and store numbers as built-in types.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is not finite or out of range.
        """
        minimum_ints = {
            "ransac_sample_points": 3,
            "ransac_max_iterations": 1,
            "max_plane_passes": 1,
        }
        for name, minimum in minimum_ints.items():
            value = _check_int(name, getattr(self, name))
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}, got {value}")
            object.__setattr__(self, name, value)
        if self.max_calibration_attempts is not None:
            attempts = _check_int(
                "max_calibration_attempts", self.max_calibration_attempts
            )
            if attempts < 1:
                raise ValueError(
                    f"max_calibration_attempts must be at least 1, got {attempts}"
                )
            object.__setattr__(self, "max_calibration_attempts", attempts)
        for name in (
            "distance_threshold_mm",
            "voxel_size_mm",
            "merge_distance_threshold_mm",
            "check_distance_mm",
        ):
            value = _check_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)
        for name in ("min_inlier_ratio", "min_on_plane_ratio"):
            ratio = _check_float(name, getattr(self, name))
            if not 0 <= ratio <= 1:
                raise ValueError(f"{name} must be from 0 to 1, got {ratio}")
            object.__setattr__(self, name, ratio)
        angle = _check_float("max_gravity_angle_deg", self.max_gravity_angle_deg)
        if not 0 <= angle < 90:
            raise ValueError(
                f"max_gravity_angle_deg must be from 0 up to 90, got {angle}"
            )
        object.__setattr__(self, "max_gravity_angle_deg", angle)
