"""Port component checks adapted from BoxLabRunners@0a2f571.

The first five cases reuse tests/test_components.py against the port's types and test-
only checked adapter. The two Runner loader cases are replaced by checks of the
production frozen binding, which deliberately has no loader, settings argument, or
optional plane estimator. Public frame/message behavior is covered separately by the
parcel node tests.
"""

import unittest
from dataclasses import replace
from unittest.mock import Mock

import numpy as np

from depthai_nodes.algorithms.parcel import create_components
from depthai_nodes.algorithms.parcel.fast_baseline_algorithm_2f65a6f6296d48e981f5fe3f77bfde89.components import (
    Components,
)
from depthai_nodes.algorithms.parcel.fast_baseline_algorithm_2f65a6f6296d48e981f5fe3f77bfde89.scene import (
    CandidateResult,
    Cuboid,
    PointCloud,
    Scene,
    Segment,
    SupportPlane,
)
from tests.parcel_reference.checked_adapter import process_scene
from tests.parcel_reference.test_fast_baseline import PARCEL, render_scene


class ComponentTests(unittest.TestCase):
    def setUp(self):
        xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
        xyz.flags.writeable = False
        self.scene = Scene(
            PointCloud(xyz, "scene"), SupportPlane((0, 0, 1), 0, "scene"), ()
        )
        self.segment = Segment(np.array([2, 0, 3], np.int32))
        self.cuboid = Cuboid((0, 0, 0), (1, 0, 0), 20, 10, 5)

    def test_one_completed_candidate_call_precedes_read_only_measurement(self):
        events = []
        candidates = [
            CandidateResult(self.segment, decision)
            for decision in (
                "background",
                "partially_visible",
                "uncertain",
                "eligible",
                "eligible",
                "eligible",
            )
        ]

        def find_candidates(scene):
            self.assertIs(scene, self.scene)
            self.assertFalse(scene.cloud.xyz_mm.flags.writeable)
            events.append("decisions completed")
            return candidates

        outcomes = iter((None, replace(self.cuboid, height_mm=None), self.cuboid))

        def measure(parcel, plane):
            self.assertEqual(events[0], "decisions completed")
            self.assertIs(plane, self.scene.plane)
            self.assertEqual(parcel.frame, self.scene.cloud.frame)
            self.assertFalse(parcel.xyz_mm.flags.writeable)
            np.testing.assert_array_equal(
                parcel.xyz_mm, self.scene.cloud.xyz_mm[[2, 0, 3]]
            )
            events.append("measure")
            return next(outcomes)

        result = process_scene(
            "mixed", self.scene, Components(find_candidates, measure), "supplied_plane"
        )
        self.assertEqual(
            events, ["decisions completed", "measure", "measure", "measure"]
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual([box["height_mm"] for box in result["detections"]], [None, 5])
        self.assertEqual(
            [box["id"] for box in result["detections"]],
            ["prediction-1", "prediction-2"],
        )

    def test_empty_rejected_and_no_fit_are_successful_empty_results(self):
        for candidates in (
            [],
            [CandidateResult(self.segment, "uncertain")],
            [CandidateResult(self.segment, "eligible")],
        ):
            with self.subTest(candidates=candidates):
                measure = Mock(return_value=None)
                result = process_scene(
                    "empty",
                    self.scene,
                    Components(lambda _, candidates=candidates: candidates, measure),
                    "supplied_plane",
                )
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["detections"], [])
                self.assertEqual(
                    measure.call_count,
                    sum(c.decision == "eligible" for c in candidates),
                )

    def test_rejects_malformed_results_and_decisions_before_measurement(self):
        invalid = [
            None,
            (),
            iter([]),
            [self.segment],
            [CandidateResult(None, "eligible")],
        ]
        invalid.extend(
            [CandidateResult(self.segment, decision)]
            for decision in (None, [], 1, "unknown")
        )
        for candidates in invalid:
            with self.subTest(candidates=candidates):
                measure = Mock()
                with self.assertRaises(ValueError):
                    process_scene(
                        "invalid",
                        self.scene,
                        Components(
                            lambda _, candidates=candidates: candidates, measure
                        ),
                        "supplied_plane",
                    )
                measure.assert_not_called()

    def test_rejects_bad_indices_even_for_rejected_candidates(self):
        invalid = [
            [],
            np.array([], np.int32),
            np.array([0], np.int64),
            np.array([[0]], np.int32),
            np.array([-1], np.int32),
            np.array([4], np.int32),
            np.array([0, 0], np.int32),
            np.array([2, 0, 2], np.int32),
        ]
        for indices in invalid:
            with self.subTest(indices=indices):
                candidates = [CandidateResult(Segment(indices), "background")]
                measure = Mock()
                with self.assertRaisesRegex(ValueError, "segment indices"):
                    process_scene(
                        "invalid",
                        self.scene,
                        Components(
                            lambda _, candidates=candidates: candidates, measure
                        ),
                        "supplied_plane",
                    )
                measure.assert_not_called()

    def test_existing_cuboid_geometry_checks_remain_enforced(self):
        invalid = [
            replace(self.cuboid, length_axis=(2, 0, 0)),
            replace(self.cuboid, length_axis=(0, 0, 1)),
            replace(self.cuboid, base_center_mm=(0, 0, 1)),
            replace(self.cuboid, width_mm=0),
            replace(self.cuboid, height_mm=float("nan")),
        ]
        for cuboid in invalid:
            with self.subTest(cuboid=cuboid), self.assertRaisesRegex(
                ValueError, "cuboid geometry"
            ):
                process_scene(
                    "geometry",
                    self.scene,
                    Components(
                        lambda _: [CandidateResult(self.segment, "eligible")],
                        lambda *_, cuboid=cuboid: cuboid,
                    ),
                    "supplied_plane",
                )

    def test_frozen_binding_rejects_settings_and_backend_arguments(self):
        for settings in ({}, {"measurement": {"min_height_mm": 1.0}}):
            with self.subTest(settings=settings), self.assertRaises(TypeError):
                create_components(settings)
        with self.assertRaises(TypeError):
            create_components(algorithm="algorithms.other")
        with self.assertRaises(TypeError):
            create_components(configuration={})

    def test_frozen_binding_completes_both_callbacks_without_estimation(self):
        scene, _ = render_scene((PARCEL,))
        scene.cloud.xyz_mm.flags.writeable = False
        for view in scene.views:
            view.image.flags.writeable = False
        components = create_components()
        self.assertIsInstance(components, Components)
        self.assertFalse(hasattr(components, "estimate_plane"))
        self.assertTrue(callable(components.find_candidates))
        self.assertTrue(callable(components.measure))
        candidates = components.find_candidates(scene)
        self.assertEqual([item.decision for item in candidates], ["eligible"])
        parcel = np.take(
            scene.cloud.xyz_mm, candidates[0].segment.point_indices, axis=0
        )
        parcel.flags.writeable = False
        cuboid = components.measure(PointCloud(parcel, scene.cloud.frame), scene.plane)
        self.assertIsNotNone(cuboid)
        np.testing.assert_allclose(
            [cuboid.length_mm, cuboid.width_mm, cuboid.height_mm],
            [180, 155, 110],
            atol=5,
        )
        self.assertFalse(scene.cloud.xyz_mm.flags.writeable)
        self.assertFalse(parcel.flags.writeable)
