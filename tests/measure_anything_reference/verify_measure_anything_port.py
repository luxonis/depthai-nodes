"""Compare the frozen MeasureAnything computation with the bundled port, without network
I/O.

The reference root must contain the pinned BoxLabRunners source and copies of the
bundled evaluation-evidence.json and scoped-settings.json named fast-baseline.json and
fast-baseline-settings.json (legacy comparison tool filenames). Optional cached inputs
must be an already-authorized validation cache from benchmark_fast.py. No labels,
training inputs, or held-out inputs are read. This is an output comparison, not a timing
benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PACKAGE = "depthai_nodes.algorithms.measure_anything.implementation"
SOURCE_COMMIT = "d5e35e17d75f91368889dc53d950792eed4096e6"
RELEASE_ID = "release-f14a597a366c46c6bdba00fa25e45bcd"
ABSOLUTE_TOLERANCE = 1e-9


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_source(root):
    bundled = REPO_ROOT.joinpath(*PACKAGE.split(".")) / "evaluation-evidence.json"
    expected = json.loads(bundled.read_bytes())
    evidence = json.loads((root / "fast-baseline.json").read_bytes())
    if expected["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Bundled source commit differs from the selected candidate")
    if evidence != expected:
        raise ValueError("Reference evidence differs from bundled frozen evidence")
    for name, expected_digest in expected["provenance"]["source_files_sha256"].items():
        if digest((root / name).read_bytes()) != expected_digest:
            raise ValueError(f"Frozen reference source changed: {name}")
    settings = (root / "fast-baseline-settings.json").read_bytes()
    if digest(settings) != expected["settings_sha256"]:
        raise ValueError("Frozen scoped settings changed")
    return evidence, json.loads(settings)


def convert_scene(scene, types):
    xyz = np.array(scene.cloud.xyz_mm, copy=True)
    xyz.flags.writeable = False
    views = []
    for view in scene.views:
        pixels = np.array(view.image, copy=True, order="K")
        pixels.flags.writeable = False
        views.append(
            types.CameraView(
                view.role,
                pixels,
                view.intrinsics,
                view.distortion_model,
                view.distortion_coefficients,
                view.scene_to_camera_mm,
            )
        )
    return types.Scene(
        types.PointCloud(xyz, scene.cloud.frame),
        types.SupportPlane(
            scene.plane.normal, scene.plane.offset_mm, scene.plane.frame
        ),
        tuple(views),
    )


def synthetic_scenes():
    from tests.measure_anything_reference.test_fast_baseline import (
        PARCEL,
        PLANE,
        render_scene,
        render_topdown,
        render_topdown_dropout,
        repeat_topdown_view,
    )

    types = importlib.import_module(PACKAGE + ".scene")
    clear, _ = render_scene((PARCEL,))
    floor, _ = render_scene()
    clipped, _ = render_scene(
        (((190, 100, 520), (440, 210, 680)),), depth_right_edge=340
    )
    occluded, _ = render_scene((PARCEL, ((-22, 70, 450), (22, 210, 480))))
    blank, _ = render_scene((PARCEL,), blank_views=("color", "left"))
    touching, _ = render_scene((PARCEL, ((90, 100, 520), (190, 210, 675))))
    multiple, _ = render_scene((PARCEL, ((150, 100, 560), (230, 210, 675))))
    dropout = render_topdown_dropout()
    uniform = repeat_topdown_view(dropout, np.full_like(dropout.views[0].image, 35))
    irregular = render_topdown(
        (((-130, -110), (-35, 110)), ((-35, 15), (130, 110))),
        contrast=150,
        background_depth=False,
    )
    strided_views = []
    for view in clear.views:
        shape = (view.image.shape[0], view.image.shape[1] * 2, *view.image.shape[2:])
        storage = np.empty(shape, dtype=np.uint8)
        pixels = storage[:, ::2]
        pixels[:] = view.image
        pixels.flags.writeable = False
        strided_views.append(replace(view, image=pixels))
    distorted = replace(
        clear,
        views=tuple(
            replace(
                view,
                distortion_model="perspective",
                distortion_coefficients=(
                    0.03,
                    -0.002,
                    0.0003,
                    -0.0002,
                    0.0001,
                    0.01,
                    -0.001,
                    0.0001,
                    0.0001,
                    0.0002,
                    -0.0001,
                    0.0002,
                    0.02,
                    -0.03,
                ),
            )
            for view in clear.views
        ),
    )
    sparse = replace(clear, cloud=types.PointCloud(clear.cloud.xyz_mm[::140], "scene"))
    return [
        ("clear", clear),
        ("one_clear_view", blank),
        ("clipped", clipped),
        ("occluded", occluded),
        ("touching", touching),
        ("multiple", multiple),
        ("floor", floor),
        (
            "empty",
            types.Scene(
                types.PointCloud(np.empty((0, 3), np.float32), "scene"),
                PLANE,
                floor.views,
            ),
        ),
        ("dropout", dropout),
        ("uniform_dropout", uniform),
        ("l_shaped", irregular),
        ("strided_images", replace(clear, views=tuple(strided_views))),
        ("distortion_tilt", distorted),
        ("sparse", sparse),
        ("repeated_clear", clear),
    ]


def cached_scenes(root):
    index = json.loads((root / "cache-index.json").read_bytes())
    manifest = json.loads((root / "manifest.json").read_bytes())
    if index["split"] != "validation" or manifest["split"] != "validation":
        raise ValueError("Only an authorized validation cache is accepted")
    if index["release_id"] != RELEASE_ID or manifest["release_id"] != RELEASE_ID:
        raise ValueError("Cached validation release differs from selected release")
    canonical = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if digest(canonical) != index["manifest_sha256"]:
        raise ValueError("Cached validation manifest changed")
    ids = [item["snapshot_id"] for item in manifest["snapshots"]]
    if set(ids) != set(index["snapshots"]) or len(ids) != len(set(ids)):
        raise ValueError("Cache index does not cover the complete validation manifest")
    types = importlib.import_module(PACKAGE + ".scene")
    scenes = []
    for snapshot_id in ids:
        if Path(snapshot_id).name != snapshot_id:
            raise ValueError("Invalid cached snapshot identifier")
        path = root / (snapshot_id + ".npz")
        if digest(path.read_bytes()) != index["snapshots"][snapshot_id]:
            raise ValueError(f"Cached validation scene changed: {snapshot_id}")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            xyz = data["xyz"]
            if xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3:
                raise ValueError("Cached cloud must contain float32 XYZ millimeters")
            if not np.isfinite(xyz).all():
                raise ValueError("Cached cloud contains nonfinite points")
            xyz.flags.writeable = False
            views = []
            for number, view in enumerate(metadata["views"]):
                image = data[f"image_{number}"]
                image.flags.writeable = False
                views.append(types.CameraView(image=image, **view))
            scene = types.Scene(
                types.PointCloud(xyz, metadata["frame"]),
                types.SupportPlane(**metadata["plane"]),
                tuple(views),
            )
        scenes.append((snapshot_id, scene))
    return scenes, index["manifest_sha256"]


def vector(cuboid):
    return np.array(
        [
            *cuboid.base_center_mm,
            *cuboid.length_axis,
            cuboid.length_mm,
            cuboid.width_mm,
            *([] if cuboid.height_mm is None else [cuboid.height_mm]),
        ]
    )


def compare_cuboid(actual, expected):
    if (actual is None) != (expected is None):
        raise AssertionError("No-fit outcomes differ")
    if actual is None:
        return 0.0
    if (actual.height_mm is None) != (expected.height_mm is None):
        raise AssertionError("Unknown-height outcomes differ")
    np.testing.assert_allclose(
        vector(actual),
        vector(expected),
        atol=ABSOLUTE_TOLERANCE,
        rtol=0,
        err_msg="Cuboid geometry differs",
    )
    return float(np.max(np.abs(vector(actual) - vector(expected))))


def compare_scene(name, scene, reference_types, reference, port, mode):
    source = convert_scene(scene, reference_types)
    scene.cloud.xyz_mm.flags.writeable = False
    for view in scene.views:
        view.image.flags.writeable = False
    before = digest(scene.cloud.xyz_mm.tobytes())
    expected = reference.find_candidates(source)
    actual = port.find_candidates(scene)
    if len(actual) != len(expected):
        raise AssertionError(f"{name}: candidate counts differ")
    maximum = 0.0
    candidates = []
    for number, (left, right) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(
            left.segment.point_indices,
            right.segment.point_indices,
            err_msg=f"{name}: candidate {number} membership/order",
        )
        if left.decision != right.decision:
            raise AssertionError(f"{name}: candidate {number} decision differs")
        left_points = np.take(scene.cloud.xyz_mm, left.segment.point_indices, axis=0)
        right_points = np.take(source.cloud.xyz_mm, right.segment.point_indices, axis=0)
        np.testing.assert_array_equal(left_points, right_points)
        left_points.flags.writeable = right_points.flags.writeable = False
        outcome = {
            "decision": left.decision,
            "indices_sha256": digest(left.segment.point_indices.tobytes()),
            "points_sha256": digest(left_points.tobytes()),
        }
        if left.decision == "eligible":
            types = importlib.import_module(PACKAGE + ".scene")
            cuboid = port.measure(
                types.PointCloud(left_points, scene.cloud.frame), scene.plane
            )
            baseline = reference.measure(
                reference_types.PointCloud(right_points, source.cloud.frame),
                source.plane,
            )
            maximum = max(maximum, compare_cuboid(cuboid, baseline))
            outcome["cuboid"] = None if cuboid is None else asdict(cuboid)
        candidates.append(outcome)
    if digest(scene.cloud.xyz_mm.tobytes()) != before:
        raise AssertionError("Scene input points changed during comparison")
    return {
        "name": name,
        "mode": mode,
        "candidate_count": len(actual),
        "eligible_count": sum(item.decision == "eligible" for item in actual),
        "maximum_geometry_absolute_difference": maximum,
        "candidates": candidates,
    }


def standalone_measurements(reference_types, reference, port):
    types = importlib.import_module(PACKAGE + ".scene")
    normal = np.array([0.0, -0.8, 0.6])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(normal, e1)
    length = 0.8 * e1 + 0.6 * e2
    width = np.cross(normal, length)
    center = 240 * e1 + 560 * e2 + 85 * normal
    plane = types.SupportPlane(tuple(normal), -85.0, "scene")
    reference_plane = reference_types.SupportPlane(
        plane.normal, plane.offset_mm, plane.frame
    )
    u, v = np.meshgrid(np.arange(-100, 101, 5), np.arange(-50, 51, 5))
    base = center + u.reshape(-1, 1) * length + v.reshape(-1, 1) * width
    top = base + 80 * normal
    outliers = np.array([center + 210 * normal, center + 500 * length + 50 * normal])
    cases = (
        ("tilted_with_outliers", np.concatenate((top, outliers))),
        ("unknown_height", base),
        ("line_no_fit", base[: u.shape[1]]),
    )
    results = []
    for name, xyz in cases:
        xyz = xyz.astype(np.float32)
        xyz.flags.writeable = False
        actual = port.measure(types.PointCloud(xyz, "scene"), plane)
        expected = reference.measure(
            reference_types.PointCloud(xyz, "scene"), reference_plane
        )
        difference = compare_cuboid(actual, expected)
        if name == "unknown_height" and (
            actual is None or actual.height_mm is not None
        ):
            raise AssertionError("Fixture did not exercise unknown height")
        if name == "line_no_fit" and actual is not None:
            raise AssertionError("Fixture did not exercise no-fit")
        results.append(
            {
                "name": name,
                "maximum_geometry_absolute_difference": difference,
                "cuboid": None if actual is None else asdict(actual),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_root = args.reference_root.resolve()
    evidence, settings = verify_source(reference_root)
    sys.path.insert(0, str(reference_root))
    reference_module = importlib.import_module(
        "algorithms.fast_baseline_oak4_visibility_01"
    )
    reference_types = importlib.import_module("runner.scene")
    binding = importlib.import_module("depthai_nodes.algorithms.measure_anything")
    if (
        binding.ALGORITHM_ID
        != evidence["validation"]["run"]["candidate"]["candidate_id"]
    ):
        raise ValueError("Port binding differs from evaluated candidate")
    scenarios = synthetic_scenes()
    cached, manifest_digest = (
        ([], None)
        if args.validation_cache is None
        else cached_scenes(args.validation_cache)
    )
    records, measurements = [], []
    reference = reference_module.create_components(settings["configuration"])
    port = binding.create_components()
    for name, scene in scenarios + cached:
        records.append(
            compare_scene(name, scene, reference_types, reference, port, "numpy")
        )
    measurements.append(
        {
            "mode": "numpy",
            "cases": standalone_measurements(reference_types, reference, port),
        }
    )
    result = {
        "schema_version": "parcel-port-comparison/1",
        "result": "passed",
        "candidate_id": binding.ALGORITHM_ID,
        "reference_source_commit": SOURCE_COMMIT,
        "reference_image": None,
        "reference_build": evidence["build"],
        "reference_source_file_count": len(
            evidence["provenance"]["source_files_sha256"]
        ),
        "scoped_settings_sha256": evidence["settings_sha256"],
        "tolerance": {
            "geometry_absolute": ABSOLUTE_TOLERANCE,
            "geometry_relative": 0,
            "indices_points_order_decisions": "exact",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
        },
        "native_available": False,
        "native_binaries_sha256": {},
        "synthetic_scene_count": len(scenarios),
        "validation_scene_count": len(cached),
        "validation_manifest_sha256": manifest_digest,
        "validation_cache_scope": "historical authorized decoded validation cache; not labels or new held-out inputs"
        if cached
        else None,
        "records": records,
        "standalone_measurements": measurements,
        "limitations": [
            "No live OAK hardware or remote serialization coverage.",
            "Comparisons share one interpreter and dependency environment per invocation.",
            "Cached scenes exercise decoded arrays, not original raw artifact decoding.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "result",
                    "synthetic_scene_count",
                    "validation_scene_count",
                    "native_available",
                    "environment",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
