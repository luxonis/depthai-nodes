"""Assess a support plane against the same limits for fit and verification."""

import math

import numpy as np

from depthai_nodes.message.plane_calibration_config import SupportPlaneCalibrationConfig
from depthai_nodes.message.support_plane import SupportPlane
from depthai_nodes.message.support_plane_calibration import (
    PlaneAssessment,
    PlaneAssessmentReason,
)


def _exceeds(value: float, limit: float) -> bool:
    # Normalizing the equation and gravity can round an inclusive limit by a few ulps.
    return value > limit and not math.isclose(value, limit, rel_tol=1e-15)


def assess_plane(
    plane: SupportPlane,
    sample: np.ndarray,
    gravity_direction: tuple[float, float, float] | np.ndarray,
    config: SupportPlaneCalibrationConfig,
) -> PlaneAssessment:
    dot = float(np.dot(plane.normal, gravity_direction))
    tilt = math.degrees(
        math.atan2(np.linalg.norm(np.cross(plane.normal, gravity_direction)), abs(dot))
    )
    intersection = -plane.offset_mm / dot if abs(dot) > 1e-12 else None
    vertical = (
        intersection
        if intersection is not None and 0 <= intersection < math.inf
        else None
    )
    distances = np.abs(sample @ plane.normal + plane.offset_mm)
    count = len(sample)
    on_plane = int(np.count_nonzero(distances <= config.check_distance_mm))

    reasons = []
    if count < config.min_sampled_point_count:
        reasons.append(
            PlaneAssessmentReason(
                code="insufficient_sample",
                measured=count,
                limit=config.min_sampled_point_count,
            )
        )
    elif on_plane / count < config.min_on_plane_ratio:
        reasons.append(
            PlaneAssessmentReason(
                code="coverage_below_minimum",
                measured=on_plane / count,
                limit=config.min_on_plane_ratio,
            )
        )
    if _exceeds(tilt, config.max_gravity_angle_deg):
        reasons.append(
            PlaneAssessmentReason(
                code="tilt_above_maximum",
                measured=tilt,
                limit=config.max_gravity_angle_deg,
            )
        )
    bounds = config.vertical_distance_range_mm
    if intersection is None or not math.isfinite(intersection):
        reasons.append(
            PlaneAssessmentReason(code="vertical_distance_unavailable", limit=bounds)
        )
    elif intersection <= 0:
        reasons.append(
            PlaneAssessmentReason(
                code="plane_not_below_camera", measured=intersection, limit=0.0
            )
        )
    if (
        vertical is not None
        and bounds is not None
        and (_exceeds(bounds[0], vertical) or _exceeds(vertical, bounds[1]))
    ):
        reasons.append(
            PlaneAssessmentReason(
                code="vertical_distance_out_of_range", measured=vertical, limit=bounds
            )
        )

    return PlaneAssessment(
        plane=plane,
        sampled_point_count=count,
        on_plane_point_count=on_plane,
        tilt_deg=tilt,
        perpendicular_distance_mm=abs(plane.offset_mm),
        vertical_distance_mm=vertical,
        reasons=tuple(reasons),
    )
