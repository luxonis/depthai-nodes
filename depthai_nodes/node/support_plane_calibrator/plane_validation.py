"""The level test shared by the candidate search and plane verification."""

import numpy as np

from .plane_calibration_config import PlaneCalibrationConfig


def orient_along_gravity(plane_eq: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """Return the plane equation with its normal pointing away from ``gravity``.

    The accelerometer vector points up, so the oriented normal points down,
    away from a camera above the plane. RANSAC and saved planes may give
    either sign; orienting them first makes `is_valid_horizontal_plane` depend
    only on the plane.

    Args:
        plane_eq: Plane ``(a, b, c, d)``.
        gravity: Raw accelerometer vector.

    Returns:
        ``plane_eq`` if its normal already points away from ``gravity``,
        otherwise the negated equation.
    """
    if np.dot(plane_eq[:3], gravity) > 0:
        return -plane_eq
    return plane_eq


def is_valid_horizontal_plane(
    plane_eq: np.ndarray, gravity: np.ndarray, config: PlaneCalibrationConfig
) -> bool:
    """Check that a plane is horizontal and lies below the camera (the level test).

    Args:
        plane_eq: Plane ``(a, b, c, d)`` with a unit normal ``(a, b, c)``,
            oriented with `orient_along_gravity`.
        gravity: Raw accelerometer vector; only its direction is used.
        config: Supplies the largest angle between the normal and the direction
            opposing gravity, and the camera height range.

    Returns:
        ``True`` if the normal is within the angle and the camera height ``-d``
        is not negative and within the configured range. ``False`` if
        ``gravity`` is the zero vector.
    """
    gravity_norm = np.linalg.norm(gravity)
    if gravity_norm == 0:
        return False
    normalized_gravity = gravity / gravity_norm
    camera_height_mm = -float(plane_eq[3])
    low, high = config.min_camera_height_mm, config.max_camera_height_mm
    return bool(
        np.dot(plane_eq[:3], normalized_gravity)
        <= -np.cos(np.deg2rad(config.max_gravity_angle_deg))
        and camera_height_mm >= (0.0 if low is None else low)
        and (high is None or camera_height_mm <= high)
    )


def gravity_angle_deg(plane_eq: np.ndarray, gravity: np.ndarray) -> float:
    """Return the angle between a plane normal and the direction opposing gravity.

    Args:
        plane_eq: Plane ``(a, b, c, d)`` with a unit normal ``(a, b, c)``.
        gravity: Raw accelerometer vector; only its direction is used.

    Returns:
        The angle in degrees, from 0 to 180.
    """
    normalized_gravity = gravity / (np.linalg.norm(gravity) + 1e-9)
    cosine = float(np.clip(-np.dot(plane_eq[:3], normalized_gravity), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
