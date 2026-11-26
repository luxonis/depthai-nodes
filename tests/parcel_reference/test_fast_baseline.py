"""Pinned BoxLabRunners fast-baseline regressions with relocated imports.

Source: luxonis/BoxLabRunners@0a2f571843f81ac6cc2f8cebe3475540e314f83d,
tests/test_fast_baseline.py. Settings changes here are upstream
regression probes, not the public frozen binding. The checked adapter is
test-only; production frame/message behavior has separate node tests.
"""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np

from depthai_nodes.algorithms.parcel import implementation
from depthai_nodes.algorithms.parcel.implementation.scene import (
    CameraView,
    CandidateResult,
    PointCloud,
    Scene,
    Segment,
    SupportPlane,
    project_points,
)
from tests.parcel_reference.checked_adapter import load_components as _load_components
from tests.parcel_reference.checked_adapter import process_scene


def load_components(configuration):
    configuration = dict(configuration, algorithm="algorithms.fast_baseline")
    return _load_components(configuration)


SETTINGS = Path(__file__).with_name("baseline-settings.json")
PLANE = SupportPlane((0.0, -1.0, 0.0), 210.0, "scene")
PARCEL = ((-90, 100, 520), (90, 210, 675))


def render_scene(boxes=(), *, depth_right_edge=None, blank_views=(), height=300):
    width = 400
    fx, fy, cx, cy = 300.0, 300.0, 199.5, 149.5
    v, u = np.indices((height, width))
    rays = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)
    intrinsics = ((fx, 0, cx), (0, fy, cy), (0, 0, 1))
    views = []
    for role, camera_x in (("color", 0), ("left", -15), ("right", 15)):
        origin = np.array([camera_x, 0, 0])
        floor_z = PLANE.offset_mm / rays[:, :, 1]
        depth = np.where((floor_z > 0) & (floor_z <= 1400), floor_z, np.inf)
        owners = np.full((height, width), -1)
        for index, (low, high) in enumerate(boxes):
            bounds = (np.asarray([low, high])[:, None, None, :] - origin) / rays
            enter = bounds.min(axis=0).max(axis=-1)
            leave = bounds.max(axis=0).min(axis=-1)
            visible = (enter > 0) & (enter <= leave) & (enter < depth)
            depth[visible] = enter[visible]
            owners[visible] = index

        image = np.where(owners < 0, 35, 185 - 60 * owners).astype(np.uint8)
        if role in blank_views:
            image.fill(35)
        transform = (
            (1, 0, 0, -camera_x),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
        if role == "color":
            image = np.repeat(image[:, :, None], 3, axis=2)
            retained = np.isfinite(depth)
            if depth_right_edge is not None:
                retained &= (owners != 0) | (u < depth_right_edge)
            cloud = PointCloud(
                (depth[retained, None] * rays[retained] + origin).astype(np.float32),
                "scene",
            )
            parcel = Segment(np.flatnonzero(owners[retained] == 0).astype(np.int32))
        views.append(CameraView(role, image, intrinsics, "none", (), transform))
    return Scene(cloud, PLANE, tuple(views)), parcel


def render_topdown(rectangles, *, contrast, background_depth):
    image_size = 260
    focal = 500.0
    center = 129.5
    intrinsics = ((focal, 0, center), (0, focal, center), (0, 0, 1))
    v, u = np.indices((image_size, image_size))
    rays = np.stack(
        ((u - center) / focal, (v - center) / focal, np.ones_like(u)), axis=-1
    )
    views = []
    for role, camera_x in (("color", 0), ("left", -15), ("right", 15)):
        top = rays[:, :, :2] * 890 + np.array([camera_x, 0])
        parcel = np.zeros(u.shape, dtype=bool)
        for (x0, y0), (x1, y1) in rectangles:
            parcel |= (
                (top[:, :, 0] >= x0)
                & (top[:, :, 0] <= x1)
                & (top[:, :, 1] >= y0)
                & (top[:, :, 1] <= y1)
            )
        image = np.where(parcel, 35 + contrast, 35).astype(np.uint8)
        if role == "color":
            image = np.repeat(image[:, :, None], 3, axis=2)
            depth = np.where(parcel, 890.0, 1000.0)
            xyz = (rays * depth[:, :, None]).astype(np.float32)
            xyz = xyz.reshape(-1, 3) if background_depth else xyz[parcel]
        transform = ((1, 0, 0, -camera_x), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        views.append(CameraView(role, image, intrinsics, "none", (), transform))
    plane = SupportPlane((0, 0, -1), 1000.0, "scene")
    return Scene(PointCloud(xyz, "scene"), plane, tuple(views))


def render_topdown_dropout():
    scene = render_topdown(
        (((-110, -80), (110, 80)),), contrast=150, background_depth=False
    )
    x, y, _ = scene.cloud.xyz_mm.T
    dropout = (x > 43) & (np.abs(y) < 20)
    return replace(
        scene, cloud=PointCloud(scene.cloud.xyz_mm[~dropout], scene.cloud.frame)
    )


def repeat_topdown_view(scene, image):
    color = scene.views[0]
    views = tuple(
        replace(
            color,
            role=view.role,
            image=image if view.role == "color" else image[:, :, 0],
        )
        for view in scene.views
    )
    return replace(scene, views=views)


def load_hull_components(max_gap=0.12):
    config = json.loads(SETTINGS.read_bytes())
    config["configuration"]["visibility"]["max_hull_gap_fraction"] = max_gap
    _, components = load_components(config)
    return components


class FrozenGeometryRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configuration = json.loads(SETTINGS.read_bytes())
        _, cls.components = load_components(configuration)
        from depthai_nodes.algorithms.parcel.implementation.visibility import (
            Visibility,
        )

        cls.visibility = Visibility(**configuration["configuration"]["visibility"])

    def test_one_clear_retained_view_is_enough(self):
        scene, parcel = render_scene((PARCEL,), blank_views=("color", "left"))
        candidates = self.components.find_candidates(scene)
        segments = [candidate.segment for candidate in candidates]
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].point_indices.dtype, np.int32)
        np.testing.assert_array_equal(
            np.unique(segments[0].point_indices), segments[0].point_indices
        )
        points = scene.cloud.xyz_mm[parcel.point_indices]
        above_plane = PLANE.offset_mm - points[:, 1] >= 18
        np.testing.assert_array_equal(
            segments[0].point_indices, parcel.point_indices[above_plane]
        )
        self.assertEqual(candidates[0].decision, "eligible")
        result = process_scene("clear", scene, self.components, "supplied_plane")
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["detections"]), 1)
        cuboid = result["detections"][0]
        np.testing.assert_allclose(
            [cuboid["length_mm"], cuboid["width_mm"], cuboid["height_mm"]],
            [180, 155, 110],
            atol=5,
        )

    def test_partial_parcel_with_in_frame_depth_is_not_measured(self):
        clipped = ((190, 100, 520), (440, 210, 680))
        scene, _ = render_scene((clipped,), depth_right_edge=340)
        candidates = self.components.find_candidates(scene)
        segments = [candidate.segment for candidate in candidates]
        self.assertEqual(len(segments), 1)
        xyz = scene.cloud.xyz_mm[segments[0].point_indices]
        cuboid = self.components.measure(
            PointCloud(xyz, scene.cloud.frame), scene.plane
        )
        self.assertIsNotNone(cuboid)
        self.assertIsNotNone(cuboid.height_mm)
        up = np.asarray(scene.plane.normal)
        long_axis = np.asarray(cuboid.length_axis)
        axes = np.stack((long_axis, np.cross(up, long_axis), up))
        unit_corners = np.array(list(product((-0.5, 0.5), (-0.5, 0.5), (0, 1))))
        size = np.array([cuboid.length_mm, cuboid.width_mm, cuboid.height_mm])
        corners = cuboid.base_center_mm + (unit_corners * size) @ axes
        for view in scene.views:
            pixels = project_points(np.concatenate((xyz, corners)), view)
            height, width = view.image.shape[:2]
            self.assertTrue(((pixels[:, 0] > 10) & (pixels[:, 0] < width - 10)).all())
            self.assertTrue(((pixels[:, 1] > 10) & (pixels[:, 1] < height - 10)).all())
            self.assertTrue((view.image[:, -1] == 185).any())

        for start, stripe_width in ((370, 0), (370, 3), (350, 3), (350, 16)):
            with self.subTest(start=start, stripe_width=stripe_width):
                striped_views = []
                for view in scene.views:
                    image = view.image.copy()
                    stripe = image[:, start : start + stripe_width]
                    stripe[stripe == 185] = 35
                    striped_views.append(replace(view, image=image))
                candidate_scene = replace(scene, views=tuple(striped_views))
                self.assertEqual(
                    self.components.find_candidates(candidate_scene)[0].decision,
                    "uncertain",
                )
                result = process_scene(
                    "clipped", candidate_scene, self.components, "supplied_plane"
                )
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["detections"], [])

    def test_occlusion_in_every_view_blocks_measurement(self):
        occluder = ((-22, 70, 450), (22, 210, 480))
        scene, parcel = render_scene((PARCEL, occluder))
        self.assertGreater(len(parcel.point_indices), 100)
        self.assertEqual(
            self.visibility.validate_candidate(scene, parcel), "partially_visible"
        )
        result = process_scene("occlusion", scene, self.components, "supplied_plane")
        self.assertEqual(result["status"], "success")
        for cuboid in result["detections"]:
            self.assertLess(cuboid["base_center_mm"][2], 480)
        self.assertLessEqual(len(result["detections"]), 1)
        framed_scene, _ = render_scene((PARCEL, occluder), height=320)
        framed_result = process_scene(
            "occlusion-with-margin", framed_scene, self.components, "supplied_plane"
        )
        self.assertEqual(framed_result["status"], "success")
        self.assertEqual(len(framed_result["detections"]), 1)
        self.assertLess(framed_result["detections"][0]["base_center_mm"][2], 480)

    def test_measured_depth_boundaries_work_in_a_uniform_image(self):
        rectangle = (((-110, -80), (110, 80)),)
        scene = render_topdown(rectangle, contrast=0, background_depth=True)
        result = process_scene("depth-only", scene, self.components, "supplied_plane")
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["detections"]), 1)
        scene = render_topdown(rectangle, contrast=0, background_depth=False)
        result = process_scene(
            "without-background", scene, self.components, "supplied_plane"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["detections"], [])

    def test_hull_recovers_a_bounded_depth_dropout_with_a_visible_outer_edge(self):
        scene = render_topdown_dropout()
        self.assertEqual(
            process_scene("dropout", scene, self.components, "supplied_plane")[
                "detections"
            ],
            [],
        )

        tight = load_hull_components(max_gap=0.01)
        self.assertEqual(
            process_scene("dropout", scene, tight, "supplied_plane")["detections"], []
        )

        recovery = load_hull_components()
        result = process_scene("dropout", scene, recovery, "supplied_plane")
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["detections"]), 1)
        cuboid = result["detections"][0]
        np.testing.assert_allclose(
            [cuboid["length_mm"], cuboid["width_mm"], cuboid["height_mm"]],
            [220, 160, 110],
            atol=5,
        )

        blank_views = tuple(
            replace(view, image=np.full_like(view.image, 35)) for view in scene.views
        )
        no_boundary = replace(scene, views=blank_views)
        self.assertEqual(
            process_scene("dropout", no_boundary, recovery, "supplied_plane")[
                "detections"
            ],
            [],
        )

    def test_hull_rejects_closer_foreground_in_the_depth_gap(self):
        scene = render_topdown_dropout()
        color = scene.views[0]
        image = color.image.copy()
        v, u = np.mgrid[126:132, 168:191]
        image[v, u] = 35
        pixels = np.column_stack((u.ravel() + 0.25, v.ravel() + 0.25, np.ones(u.size)))
        foreground = 800 * pixels @ np.linalg.inv(color.intrinsics).T
        xyz = np.concatenate((scene.cloud.xyz_mm, foreground.astype(np.float32)))
        scene = repeat_topdown_view(
            replace(scene, cloud=PointCloud(xyz, scene.cloud.frame)), image
        )

        result = process_scene(
            "occluded-dropout", scene, load_hull_components(), "supplied_plane"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["detections"], [])

    def test_hull_rejects_a_continuous_strip_to_the_frame(self):
        scene = render_topdown_dropout()
        image = scene.views[0].image.copy()
        image[120:138, 180:] = 185
        clipped = repeat_topdown_view(scene, image)
        recovery = load_hull_components()
        result = process_scene("clipped-dropout", clipped, recovery, "supplied_plane")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["detections"], [])

        separated = image.copy()
        separated[120:138, 192:-10] = 35
        enclosed = repeat_topdown_view(scene, separated)
        result = process_scene("enclosed-dropout", enclosed, recovery, "supplied_plane")
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["detections"]), 1)

    def test_visible_l_shaped_parcel_uses_its_observed_silhouette(self):
        rectangles = (((-130, -110), (-35, 110)), ((-35, 15), (130, 110)))
        scene = render_topdown(rectangles, contrast=150, background_depth=False)
        result = process_scene("irregular", scene, self.components, "supplied_plane")
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["detections"]), 1)
        cuboid = result["detections"][0]
        np.testing.assert_allclose(
            [cuboid["length_mm"], cuboid["width_mm"], cuboid["height_mm"]],
            [260, 220, 110],
            atol=5,
        )

    def test_local_measurement_quantiles_preserve_l_shaped_parcel_limits(self):
        rectangles = (((-130, -110), (-35, 110)), ((-35, 15), (130, 110)))
        scene = render_topdown(rectangles, contrast=150, background_depth=False)
        config = json.loads(SETTINGS.read_bytes())
        config["configuration"]["measurement"].update(
            footprint_edge_quantile=0.55,
            height_cell_quantile=0.5,
            extent_quantile_window_mm=25.0,
        )
        _, components = load_components(config)

        result = process_scene("irregular", scene, components, "supplied_plane")

        self.assertEqual(len(result["detections"]), 1)
        cuboid = result["detections"][0]
        np.testing.assert_allclose(
            [cuboid["length_mm"], cuboid["width_mm"], cuboid["height_mm"]],
            [260, 220, 110],
            atol=5,
        )

    def test_separated_extents_keep_a_supported_tab_and_raised_step(self):
        body = np.array(
            list(product(range(-100, 81, 2), range(-80, 81, 2))), dtype=np.float32
        )
        tab = np.array(
            list(product(range(82, 101, 2), range(-16, 17, 2))), dtype=np.float32
        )
        with_tab = np.vstack((body, tab))
        step_heights = np.where((np.abs(body) <= 12).all(axis=1), 120.0, 100.0)
        plane = SupportPlane((0.0, 0.0, 1.0), 0.0, "scene")

        config = json.loads(SETTINGS.read_bytes())
        measurement = config["configuration"]["measurement"]
        measurement.update(
            footprint_edge_quantile=0.55,
            height_cell_quantile=0.5,
            extent_quantile_window_mm=25.0,
        )
        _, localized = load_components(config)
        measurement["preserve_separated_extents"] = True
        _, preserving = load_components(config)

        for name, xy, height, expected in (
            ("edge tab", with_tab, np.full(len(with_tab), 100.0), [200, 160, 100]),
            ("raised step", body, step_heights, [180, 160, 120]),
        ):
            with self.subTest(shape=name):
                cloud = PointCloud(
                    np.column_stack((xy, height)).astype(np.float32), "scene"
                )
                trimmed = localized.measure(cloud, plane)
                cuboid = preserving.measure(cloud, plane)
                self.assertIsNotNone(trimmed)
                self.assertIsNotNone(cuboid)
                np.testing.assert_allclose(
                    [trimmed.length_mm, trimmed.width_mm, trimmed.height_mm],
                    [180, 160, 100],
                    atol=0.01,
                )
                np.testing.assert_allclose(
                    [cuboid.length_mm, cuboid.width_mm, cuboid.height_mm],
                    expected,
                    atol=0.01,
                )

    def test_measured_background_supports_a_low_contrast_edge(self):
        scene, parcel = render_scene((PARCEL,), blank_views=("left", "right"))
        color, left, right = scene.views
        image = color.image.copy()
        for row in image:
            parcel_pixels = np.flatnonzero(row[:, 0] == 185)
            if len(parcel_pixels):
                end = parcel_pixels[-1] + 1
                row[end : end + 40] = 185
        scene = replace(scene, views=(replace(color, image=image), left, right))
        self.assertEqual(self.visibility.validate_candidate(scene, parcel), "eligible")

        xyz = scene.cloud.xyz_mm[parcel.point_indices]
        parcel_only = replace(scene, cloud=PointCloud(xyz, scene.cloud.frame))
        candidate = Segment(np.arange(len(xyz), dtype=np.int32))
        self.assertEqual(
            self.visibility.validate_candidate(parcel_only, candidate), "uncertain"
        )

    def test_touching_parcels_merge_in_detection(self):
        neighbor = ((90, 100, 520), (190, 210, 675))
        scene, _ = render_scene((PARCEL, neighbor))
        candidates = self.components.find_candidates(scene)
        segments = [candidate.segment for candidate in candidates]
        self.assertEqual(len(segments), 1)
        points = scene.cloud.xyz_mm[segments[0].point_indices]
        self.assertTrue((points[:, 0] < 0).any())
        self.assertTrue((points[:, 0] > 150).any())

    def test_sparse_depth_leaves_visibility_uncertain(self):
        scene, parcel = render_scene((PARCEL,))
        sparse = scene.cloud.xyz_mm[parcel.point_indices[::140]]
        cloud = PointCloud(sparse, "scene")
        candidate = Segment(np.arange(len(sparse), dtype=np.int32))
        self.assertEqual(
            self.visibility.validate_candidate(
                Scene(cloud, PLANE, scene.views), candidate
            ),
            "uncertain",
        )

    def test_empty_and_plane_only_scenes(self):
        floor, _ = render_scene()
        empty_cloud = PointCloud(np.empty((0, 3), dtype=np.float32), "scene")
        for scene in (floor, Scene(empty_cloud, PLANE, floor.views)):
            with self.subTest(point_count=len(scene.cloud.xyz_mm)):
                result = process_scene(
                    "empty", scene, self.components, "supplied_plane"
                )
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["detections"], [])

    def test_tilted_footprint_and_height_ignore_isolated_outliers(self):
        normal = np.array([0.0, -0.8, 0.6])
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.cross(normal, e1)
        length = 0.8 * e1 + 0.6 * e2
        width = np.cross(normal, length)
        center = 240 * e1 + 560 * e2 + 85 * normal
        plane = SupportPlane(tuple(normal), -85.0, "scene")
        u, v = np.meshgrid(np.arange(-100, 101, 5), np.arange(-50, 51, 5))
        base = center + u.reshape(-1, 1) * length + v.reshape(-1, 1) * width
        top = base + 80 * normal
        outliers = np.array(
            [center + 210 * normal, center + 500 * length + 50 * normal]
        )
        cloud = PointCloud(np.concatenate((top, outliers)).astype(np.float32), "scene")
        cuboid = self.components.measure(cloud, plane)
        self.assertIsNotNone(cuboid)
        np.testing.assert_allclose(cuboid.base_center_mm, center, atol=1e-3)
        self.assertAlmostEqual(
            np.dot(cuboid.base_center_mm, normal) + plane.offset_mm, 0.0
        )
        self.assertAlmostEqual(abs(np.dot(cuboid.length_axis, length)), 1.0)
        self.assertAlmostEqual(np.dot(cuboid.length_axis, normal), 0.0)
        self.assertAlmostEqual(np.linalg.norm(cuboid.length_axis), 1.0)
        np.testing.assert_allclose(
            [cuboid.length_mm, cuboid.width_mm, cuboid.height_mm],
            [200, 100, 80],
            atol=1e-3,
        )
        footprint = self.components.measure(
            PointCloud(base.astype(np.float32), "scene"), plane
        )
        self.assertIsNotNone(footprint)
        self.assertIsNone(footprint.height_mm)
        line = PointCloud(base[: u.shape[1]].astype(np.float32), "scene")
        self.assertIsNone(self.components.measure(line, plane))

    def test_clean_large_footprint_keeps_supported_extrema(self):
        x, z = np.meshgrid(np.arange(-350, 351), np.arange(520, 821))
        xyz = np.column_stack((x.ravel(), np.full(x.size, 100), z.ravel())).astype(
            np.float32
        )
        cuboid = self.components.measure(PointCloud(xyz, "scene"), PLANE)
        self.assertIsNotNone(cuboid)
        np.testing.assert_allclose(
            [cuboid.length_mm, cuboid.width_mm, cuboid.height_mm],
            [700, 300, 110],
            atol=0.01,
        )


class CandidateCallbackTests(unittest.TestCase):
    def configuration(self, parallel=False):
        settings = Path(implementation.__file__).with_name("scoped-settings.json")
        configuration = json.loads(settings.read_text(encoding="utf-8"))[
            "configuration"
        ]
        # These callback mocks return indices into the full fixture cloud.
        # Disable sampling only for these isolated callback regression probes.
        configuration.setdefault("pipeline", {}).update(
            parallel_validation=parallel, point_stride=1, fallback_point_stride=1
        )
        return configuration

    def test_returns_ordered_decisions_without_measurement(self):
        from depthai_nodes.algorithms.parcel.implementation.factory import (
            create_components,
        )
        from depthai_nodes.algorithms.parcel.implementation.visibility import (
            Visibility,
        )

        scene, parcel = render_scene((PARCEL,))
        segments = [
            Segment(part.astype(np.int32))
            for part in np.array_split(parcel.point_indices, 2)
        ]
        for parallel in (False, True):
            with self.subTest(parallel=parallel):

                def decision(scene, segment):
                    return "eligible" if segment is segments[0] else "partially_visible"

                with (
                    patch(
                        "depthai_nodes.algorithms.parcel.implementation.factory.detect",
                        return_value=segments,
                    ),
                    patch.object(
                        Visibility, "validate_candidate", side_effect=decision
                    ) as validate,
                    patch(
                        "depthai_nodes.algorithms.parcel.implementation.factory.measure"
                    ) as measure,
                ):
                    components = create_components(self.configuration(parallel))
                    results = components.find_candidates(scene)
                self.assertEqual([result.segment for result in results], segments)
                self.assertEqual(
                    [result.decision for result in results],
                    ["eligible", "partially_visible"],
                )
                self.assertTrue(
                    all(isinstance(result, CandidateResult) for result in results)
                )
                self.assertEqual(validate.call_count, 2)
                measure.assert_not_called()

    def test_reused_scene_does_not_reuse_projection_or_decision_caches(self):
        from depthai_nodes.algorithms.parcel.implementation.factory import (
            create_components,
        )

        scene, _ = render_scene((PARCEL,))
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                reused = create_components(self.configuration(parallel))
                original = scene.cloud.xyz_mm.copy()
                before = reused.find_candidates(scene)
                self.assertIn("eligible", [result.decision for result in before])
                scene.cloud.xyz_mm[:, 0] += 10000
                try:
                    actual = reused.find_candidates(scene)
                    expected = create_components(
                        self.configuration(parallel)
                    ).find_candidates(scene)
                    self.assertEqual(
                        [result.decision for result in actual],
                        [result.decision for result in expected],
                    )
                    self.assertNotIn("eligible", [result.decision for result in actual])
                    for result, reference in zip(actual, expected):
                        np.testing.assert_array_equal(
                            result.segment.point_indices,
                            reference.segment.point_indices,
                        )
                finally:
                    scene.cloud.xyz_mm[:] = original
                after = reused.find_candidates(scene)
                self.assertEqual(
                    [result.decision for result in after],
                    [result.decision for result in before],
                )
                for result, reference in zip(after, before):
                    np.testing.assert_array_equal(
                        result.segment.point_indices, reference.segment.point_indices
                    )

    def test_projection_workers_finish_before_return_or_failure(self):
        from depthai_nodes.algorithms.parcel.implementation.factory import (
            create_components,
        )
        from depthai_nodes.algorithms.parcel.implementation.visibility import (
            Visibility,
        )

        scene, parcel = render_scene((PARCEL,))
        for path in ("empty", "overlap", "detect_failure", "prepare_failure"):
            with self.subTest(path=path):
                started, release, detected, finished = (Event() for _ in range(4))
                waiting, exited, wait_or_exit = (Event() for _ in range(3))
                completed = []
                components = create_components(self.configuration())

                class ProjectionExecutor(ThreadPoolExecutor):
                    wait_started = waiting
                    wait_observed = wait_or_exit

                    def submit(self, *arguments, **keywords):
                        future = super().submit(*arguments, **keywords)
                        result = future.result

                        def wait_for_result(*arguments, **keywords):
                            self.wait_started.set()
                            self.wait_observed.set()
                            return result(*arguments, **keywords)

                        future.result = wait_for_result
                        return future

                    def shutdown(self, wait=True, **keywords):
                        if wait:
                            self.wait_started.set()
                            self.wait_observed.set()
                        return super().shutdown(wait=wait, **keywords)

                def project(
                    *arguments,
                    started=started,
                    release=release,
                    completed=completed,
                    finished=finished,
                ):
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError("Projection test was not released")
                    completed.append(arguments[1].role)
                    if len(completed) == len(scene.views):
                        finished.set()
                    return None

                def detect(
                    *arguments,
                    started=started,
                    detected=detected,
                    path=path,
                    **keywords,
                ):
                    if not started.wait(5):
                        raise RuntimeError("Projection test did not start")
                    detected.set()
                    if path == "detect_failure":
                        raise ValueError("Detection failed")
                    return [parcel, parcel] if path == "overlap" else []

                def invoke(
                    path=path,
                    components=components,
                    finished=finished,
                    exited=exited,
                    wait_or_exit=wait_or_exit,
                ):
                    try:
                        if path in {"detect_failure", "prepare_failure"}:
                            with self.assertRaisesRegex(ValueError, "failed"):
                                components.find_candidates(scene)
                        else:
                            components.find_candidates(scene)
                    finally:
                        exited.set()
                        wait_or_exit.set()
                    self.assertTrue(finished.is_set())

                with (
                    patch(
                        "depthai_nodes.algorithms.parcel.implementation.factory.ThreadPoolExecutor",
                        ProjectionExecutor,
                    ),
                    patch(
                        "depthai_nodes.algorithms.parcel.implementation.numpy_ops.project_cells",
                        side_effect=project,
                    ),
                    patch(
                        "depthai_nodes.algorithms.parcel.implementation.factory.detect",
                        side_effect=detect,
                    ),
                    patch.object(
                        Visibility, "validate_candidate", return_value="uncertain"
                    ),
                    ThreadPoolExecutor(max_workers=1) as caller,
                ):
                    if path == "prepare_failure":
                        preparation = patch.object(
                            Visibility,
                            "prepare",
                            side_effect=ValueError("Preparation failed"),
                        )
                    else:
                        preparation = patch.object(
                            Visibility, "prepare", Visibility.prepare
                        )
                    with preparation:
                        pending = caller.submit(invoke)
                        try:
                            self.assertTrue(detected.wait(5))
                            self.assertTrue(wait_or_exit.wait(5))
                            self.assertTrue(waiting.is_set())
                            self.assertFalse(exited.is_set())
                        finally:
                            release.set()
                        pending.result(timeout=5)
                        self.assertEqual(sorted(completed), ["color", "left", "right"])


class AdapterIndexRegressionTests(unittest.TestCase):
    def test_unsorted_unique_indices_keep_parcel_order(self):
        from depthai_nodes.algorithms.parcel.implementation.components import (
            Components,
        )

        scene, _ = render_scene((PARCEL,))
        indices = np.array([5, 1, 3], np.int32)
        parcels = []

        def fit(parcel, plane):
            parcels.append(parcel.xyz_mm.copy())
            self.assertFalse(parcel.xyz_mm.flags.writeable)
            return None

        c = Components(lambda _: [CandidateResult(Segment(indices), "eligible")], fit)
        process_scene("ordering", scene, c, "supplied_plane")
        np.testing.assert_array_equal(parcels[0], scene.cloud.xyz_mm[indices])

    def test_adjacent_and_nonadjacent_duplicates_are_rejected(self):
        from depthai_nodes.algorithms.parcel.implementation.components import (
            Components,
        )

        scene, _ = render_scene((PARCEL,))
        for values in ([1, 1, 3], [1, 3, 1], [3, 1, 3]):
            with self.subTest(values=values):
                c = Components(
                    lambda _, values=values: [
                        CandidateResult(Segment(np.array(values, np.int32)), "eligible")
                    ],
                    lambda *_: None,
                )
                with self.assertRaisesRegex(ValueError, "segment indices"):
                    process_scene("duplicates", scene, c, "supplied_plane")


if __name__ == "__main__":
    unittest.main()
