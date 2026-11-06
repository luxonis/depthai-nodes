import numpy as np
import pytest

from depthai_nodes.node.support_plane_calibrator.plane_validation import (
    gravity_angle_deg,
    is_valid_horizontal_plane,
    orient_along_gravity,
)

GRAVITY = np.array([0.0, 0.0, -9.8])


def tilted(angle_deg, offset_mm=-1000.0):
    angle = np.deg2rad(angle_deg)
    return np.array([0.0, np.sin(angle), np.cos(angle), offset_mm])


@pytest.mark.parametrize("angle", [0, 3, 37.5, 90, 180])
def test_gravity_angle_is_measured_from_the_direction_opposing_gravity(angle):
    assert gravity_angle_deg(tilted(angle), GRAVITY) == pytest.approx(angle, abs=1e-2)


def test_gravity_magnitude_does_not_matter():
    assert gravity_angle_deg(tilted(10), GRAVITY * 3) == pytest.approx(10, abs=1e-4)


def test_accepts_planes_within_the_tolerance():
    assert is_valid_horizontal_plane(tilted(6), GRAVITY, 6.5)
    assert not is_valid_horizontal_plane(tilted(7), GRAVITY, 6.5)


def test_rejects_a_plane_behind_the_camera():
    assert not is_valid_horizontal_plane(tilted(0, offset_mm=1000), GRAVITY, 6.5)


def test_rejects_the_opposite_sign_equation_of_an_accepted_plane():
    assert is_valid_horizontal_plane(np.array([0, 0, 1, -1000.0]), GRAVITY, 6.5)
    assert not is_valid_horizontal_plane(np.array([0, 0, -1, 1000.0]), GRAVITY, 6.5)


def test_rejects_any_plane_when_gravity_is_zero():
    assert not is_valid_horizontal_plane(tilted(0), np.zeros(3), 90)


def test_orienting_flips_a_normal_that_points_up():
    oriented = orient_along_gravity(np.array([0, 0, -1, 1000.0]), GRAVITY)

    np.testing.assert_array_equal(oriented, [0, 0, 1, -1000.0])


def test_orienting_keeps_a_normal_that_points_down():
    plane = tilted(30)

    np.testing.assert_array_equal(orient_along_gravity(plane, GRAVITY), plane)
