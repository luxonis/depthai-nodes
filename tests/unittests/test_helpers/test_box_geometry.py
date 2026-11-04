import numpy as np
import pytest

pytest.importorskip("open3d")

from depthai_nodes.node.box_dimensioner.box_geometry import BoxGeometry  # noqa: E402

FLOOR = np.array([0.0, 0.0, 1.0, -1000.0])
# Voxel downsampling and outlier removal trim a few millimeters off each edge.
EDGE_TOLERANCE_MM = 10


def box_top(length=200.0, width=100.0, height=50.0, step=2.0, angle=0.0):
    """Points on the top of a box standing on a floor 1000 mm from the camera."""
    u, v = np.meshgrid(
        np.arange(-length / 2, length / 2 + 1e-9, step),
        np.arange(-width / 2, width / 2 + 1e-9, step),
    )
    cos, sin = np.cos(angle), np.sin(angle)
    x = cos * u.ravel() - sin * v.ravel()
    y = sin * u.ravel() + cos * v.ravel()
    return np.column_stack((x, y, np.full(x.size, 1000.0 - height)))


def test_measures_a_flat_topped_box():
    dimensions = BoxGeometry().measure(box_top(), FLOOR)

    assert dimensions is not None
    assert dimensions.height_mm == pytest.approx(50, abs=0.5)
    assert dimensions.length_mm == pytest.approx(200, abs=EDGE_TOLERANCE_MM)
    assert dimensions.width_mm == pytest.approx(100, abs=EDGE_TOLERANCE_MM)


def test_rotated_box_has_the_same_footprint():
    dimensions = BoxGeometry().measure(box_top(angle=np.pi / 6), FLOOR)

    assert dimensions is not None
    assert dimensions.length_mm == pytest.approx(200, abs=EDGE_TOLERANCE_MM)
    assert dimensions.width_mm == pytest.approx(100, abs=EDGE_TOLERANCE_MM)


def test_plane_scale_and_sign_do_not_change_the_result():
    expected = BoxGeometry().measure(box_top(), FLOOR)

    assert BoxGeometry().measure(box_top(), -3 * FLOOR) == expected


def test_too_few_points_give_no_dimensions():
    assert BoxGeometry().measure(box_top()[:49], FLOOR) is None


def test_box_flush_with_the_plane_has_no_dimensions():
    assert BoxGeometry().measure(box_top(height=0), FLOOR) is None


def test_zero_plane_normal_is_rejected():
    with pytest.raises(ValueError):
        BoxGeometry().measure(box_top(), np.array([0.0, 0.0, 0.0, -1000.0]))
