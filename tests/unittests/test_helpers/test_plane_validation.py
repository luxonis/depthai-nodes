import numpy as np
import pytest

from depthai_nodes.node import PlaneCalibrationConfig
from depthai_nodes.node.support_plane_calibrator.plane_validation import (
    gravity_angle_deg,
    is_valid_horizontal_plane,
    orient_along_gravity,
)

GRAVITY = np.array([0.0, 0.0, -9.8])
LEVEL_WITHIN_6_5 = PlaneCalibrationConfig(max_gravity_angle_deg=6.5)


def tilted(angle_deg, offset_mm=-1000.0):
    angle = np.deg2rad(angle_deg)
    return np.array([0.0, np.sin(angle), np.cos(angle), offset_mm])


@pytest.mark.parametrize("angle", [0, 3, 37.5, 90, 180])
def test_gravity_angle_is_measured_from_the_direction_opposing_gravity(angle):
    assert gravity_angle_deg(tilted(angle), GRAVITY) == pytest.approx(angle, abs=1e-2)


def test_gravity_magnitude_does_not_matter():
    assert gravity_angle_deg(tilted(10), GRAVITY * 3) == pytest.approx(10, abs=1e-4)


def test_accepts_planes_within_the_tolerance():
    assert is_valid_horizontal_plane(tilted(6), GRAVITY, LEVEL_WITHIN_6_5)
    assert not is_valid_horizontal_plane(tilted(7), GRAVITY, LEVEL_WITHIN_6_5)


def test_rejects_a_plane_behind_the_camera():
    assert not is_valid_horizontal_plane(
        tilted(0, offset_mm=1000), GRAVITY, LEVEL_WITHIN_6_5
    )


@pytest.mark.parametrize(
    ("low", "high", "offset_mm", "passes"),
    [
        (900, 1100, -899, False),
        (900, 1100, -900, True),
        (900, 1100, -1100, True),
        (900, 1100, -1101, False),
        (900, None, -5000, True),
        (None, 1100, -1, True),
        (None, 1100, 1, False),
    ],
    ids=[
        "below-min",
        "at-min",
        "at-max",
        "above-max",
        "no-max",
        "no-min",
        "no-min-behind-camera",
    ],
)
def test_camera_height_must_be_within_the_configured_range(
    low, high, offset_mm, passes
):
    config = PlaneCalibrationConfig(
        max_gravity_angle_deg=6.5,
        min_camera_height_mm=low,
        max_camera_height_mm=high,
    )

    assert is_valid_horizontal_plane(tilted(0, offset_mm), GRAVITY, config) is passes


def test_rejects_the_opposite_sign_equation_of_an_accepted_plane():
    accepted = np.array([0, 0, 1, -1000.0])

    assert is_valid_horizontal_plane(accepted, GRAVITY, LEVEL_WITHIN_6_5)
    assert not is_valid_horizontal_plane(-accepted, GRAVITY, LEVEL_WITHIN_6_5)


def test_rejects_any_plane_when_gravity_is_zero():
    config = PlaneCalibrationConfig(max_gravity_angle_deg=89)

    assert not is_valid_horizontal_plane(tilted(0), np.zeros(3), config)


def test_orienting_flips_a_normal_that_points_up():
    oriented = orient_along_gravity(np.array([0, 0, -1, 1000.0]), GRAVITY)

    np.testing.assert_array_equal(oriented, [0, 0, 1, -1000.0])


def test_orienting_keeps_a_normal_that_points_down():
    plane = tilted(30)

    np.testing.assert_array_equal(orient_along_gravity(plane, GRAVITY), plane)
