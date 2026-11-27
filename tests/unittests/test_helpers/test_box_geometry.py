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


def close_pairs():
    """``MIN_POINTS`` points on a box top in pairs closer than one voxel."""
    pairs = -(-BoxGeometry.MIN_POINTS // 2)
    side = int(np.ceil(np.sqrt(pairs)))
    grid = np.arange(side) * 20.0
    centers = np.array([[x, y, 950.0] for x in grid for y in grid])[:pairs]
    nudge = np.array([BoxGeometry.VOXEL_SIZE / 10, 0.0, 0.0])
    return np.vstack((centers, centers + nudge))[: BoxGeometry.MIN_POINTS]


def clean(points):
    return BoxGeometry().clean(points)


def test_clean_removes_an_isolated_outlier():
    outlier = np.array([[0.0, 0.0, 0.0]])

    cleaned = clean(np.vstack((box_top(), outlier)))

    assert cleaned.shape[0] > 0
    distances = np.linalg.norm(cleaned - outlier, axis=1)
    assert distances.min() > BoxGeometry.VOXEL_SIZE


def test_clean_keeps_at_most_one_point_per_voxel():
    voxel = BoxGeometry.VOXEL_SIZE
    # Clusters a tenth of a voxel wide, two voxels apart, each inside one voxel.
    grid = np.arange(0, 20 * voxel, 2 * voxel)
    centers = np.array([[x, y, 950.0] for x in grid for y in grid])
    offsets = np.array(
        [[dx, dy, 0.0] for dx in (0, voxel / 10) for dy in (0, voxel / 10)]
    )
    points = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)

    cleaned = clean(points)

    assert cleaned.shape[0] > 0
    distances = np.linalg.norm(cleaned[:, None, :] - centers[None, :, :], axis=2)
    assert np.all(distances.min(axis=1) < voxel)
    assert np.all(np.bincount(distances.argmin(axis=1)) <= 1)


def test_clean_of_no_points_is_empty():
    assert clean(np.empty((0, 3))).shape == (0, 3)


def test_measure_does_not_clean_its_points():
    points = close_pairs()
    assert clean(points).shape[0] < BoxGeometry.MIN_POINTS

    assert BoxGeometry().measure(points, FLOOR) is not None


def test_measures_a_flat_topped_box():
    dimensions = BoxGeometry().measure(clean(box_top()), FLOOR)

    assert dimensions is not None
    assert dimensions.height_mm == pytest.approx(50, abs=0.5)
    assert dimensions.length_mm == pytest.approx(200, abs=EDGE_TOLERANCE_MM)
    assert dimensions.width_mm == pytest.approx(100, abs=EDGE_TOLERANCE_MM)


def test_rotated_box_has_the_same_footprint():
    dimensions = BoxGeometry().measure(clean(box_top(angle=np.pi / 6)), FLOOR)

    assert dimensions is not None
    assert dimensions.length_mm == pytest.approx(200, abs=EDGE_TOLERANCE_MM)
    assert dimensions.width_mm == pytest.approx(100, abs=EDGE_TOLERANCE_MM)


def test_plane_scale_and_sign_do_not_change_the_result():
    expected = BoxGeometry().measure(clean(box_top()), FLOOR)

    assert BoxGeometry().measure(clean(box_top()), -3 * FLOOR) == expected


def test_too_few_points_give_no_dimensions():
    assert BoxGeometry().measure(close_pairs()[:-1], FLOOR) is None


def test_box_flush_with_the_plane_has_no_dimensions():
    assert BoxGeometry().measure(clean(box_top(height=0)), FLOOR) is None


def test_zero_plane_normal_is_rejected():
    with pytest.raises(ValueError):
        BoxGeometry().measure(clean(box_top()), np.array([0.0, 0.0, 0.0, -1000.0]))
