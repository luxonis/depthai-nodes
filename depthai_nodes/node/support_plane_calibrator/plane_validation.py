"""Gravity check shared by the candidate search and plane verification."""

import numpy as np


def is_valid_horizontal_plane(
    plane_eq: np.ndarray, gravity: np.ndarray, max_gravity_angle_deg: float
) -> bool:
    """Check that a plane faces against gravity and lies in front of the camera.

    Args:
        plane_eq: Plane ``(a, b, c, d)`` with a unit normal ``(a, b, c)``.
        gravity: Raw accelerometer vector; only its direction is used.
        max_gravity_angle_deg: Largest accepted angle between the normal and
            the direction opposing gravity.

    Returns:
        ``True`` if the normal is within the angle and the offset ``d`` is not
        positive.
    """
    normalized_gravity = gravity / (np.linalg.norm(gravity) + 1e-9)
    return bool(
        np.dot(plane_eq[:3], normalized_gravity)
        <= -np.cos(np.deg2rad(max_gravity_angle_deg))
        and plane_eq[3] <= 0
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
