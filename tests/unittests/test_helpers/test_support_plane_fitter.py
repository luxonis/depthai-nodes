from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("open3d")

from depthai_nodes.node import PlaneCalibrationConfig  # noqa: E402
from depthai_nodes.node.support_plane_calibrator.support_plane_fitter import (  # noqa: E402
    SupportPlaneFitter,
)
from tests.utils.scripted_open3d import ScriptedScene  # noqa: E402

GRAVITY = np.array([0.0, 0.0, -9.8])


def fitter(**config):
    return SupportPlaneFitter((20, 20), PlaneCalibrationConfig(**config))


def flat_cloud(depth=1000.0):
    x, y = np.meshgrid(np.arange(20) * 40.0, np.arange(20) * 40.0)
    return np.column_stack([x.ravel(), y.ravel(), np.full(400, depth)])


def offsets(scene, **config):
    with patch("open3d.geometry.PointCloud", scene.cloud):
        candidates = fitter(**config).find_candidates(flat_cloud(), GRAVITY)
    return sorted(candidate.plane.offset_mm for candidate in candidates)


def test_finds_a_floor_facing_against_gravity():
    candidates = fitter().find_candidates(flat_cloud(), GRAVITY)

    assert candidates[0].plane.offset_mm == pytest.approx(-1000, abs=1)
    assert candidates[0].plane.normal[2] == pytest.approx(1, abs=0.001)


def test_cloud_without_depth_has_no_candidates():
    assert fitter().find_candidates(flat_cloud(depth=0), GRAVITY) == []


def test_rejects_a_cloud_that_is_not_organized_to_its_size():
    with pytest.raises(ValueError, match="not organized"):
        fitter().find_candidates(np.ones((10, 3)), GRAVITY)


def test_near_parallel_planes_through_one_surface_count_once():
    scene = ScriptedScene(
        [
            ([0.0, 0.0, 1.0, -1500.0], 100),
            ([0.0, 0.0, 1.0, -1000.0], 150),
            ([0.0, 0.0, 1.0, -1010.0], 100),
        ]
    )

    assert offsets(scene, max_plane_passes=3) == [-1500, -1000]


def test_merged_near_parallel_plane_reports_only_its_own_inliers():
    # Two thirds of the points lie at 1000 mm and one third at 1012 mm. With a
    # 4 mm RANSAC threshold each layer is its own plane, and the two merge.
    cloud = flat_cloud()
    cloud[np.arange(400) % 3 == 2, 2] = 1012.0

    candidates = fitter(distance_threshold_mm=4, max_plane_passes=3).find_candidates(
        cloud, GRAVITY
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.plane.offset_mm == pytest.approx(-1000, abs=1)
    # A merge that summed both layers' inliers would give a ratio of 1.
    assert candidate.inlier_ratio == pytest.approx(2 / 3, abs=0.05)


def test_floor_fitted_with_the_opposite_sign_is_oriented_along_gravity():
    scene = ScriptedScene([([0.0, 0.0, -1.0, 1000.0], 100)])

    with patch("open3d.geometry.PointCloud", scene.cloud):
        candidates = fitter().find_candidates(flat_cloud(), GRAVITY)

    assert [(c.plane.normal, c.plane.offset_mm) for c in candidates] == [
        ((0.0, 0.0, 1.0), -1000.0)
    ]


def test_tilted_planes_or_planes_above_the_camera_are_dropped():
    scene = ScriptedScene(
        [
            # Above the camera, in both signs.
            ([0.0, 0.0, -1.0, -1000.0], 100),
            ([0.0, 0.0, 1.0, 1000.0], 100),
            # Vertical.
            ([1.0, 0.0, 0.0, -1000.0], 100),
            ([0.0, 0.0, 1.0, -1200.0], 100),
        ]
    )

    assert offsets(scene) == [-1200]


def test_stops_after_the_configured_number_of_planes():
    scene = ScriptedScene(
        [([0.0, 0.0, 1.0, -1000.0 - 500.0 * i], 50) for i in range(4)]
    )

    assert offsets(scene, max_plane_passes=2) == [-1500, -1000]


def test_stops_at_a_plane_with_too_few_inliers():
    scene = ScriptedScene(
        [([0.0, 0.0, 1.0, -1000.0], 100), ([0.0, 0.0, 1.0, -1500.0], 10)]
    )

    assert offsets(scene, min_inlier_ratio=0.1) == [-1000]


def test_fit_errors_propagate():
    scene = ScriptedScene([], error=ValueError("broken geometry"))

    with pytest.raises(ValueError, match="broken geometry"):
        offsets(scene)
