"""Test-only checked adapter from BoxLabRunners@0a2f571.

This preserves the upstream regression harness. It is not installed runtime code and
does not replace tests of the public parcel node boundary.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from depthai_nodes.algorithms.parcel.fast_baseline_algorithm_2f65a6f6296d48e981f5fe3f77bfde89.scene import (
    CandidateResult,
    Cuboid,
    PlaneEstimator,
    PointCloud,
    Scene,
    Segment,
    SupportPlane,
)

MAX_RESULT_COORDINATE_MM = 10_000_000


@dataclass(frozen=True)
class Components:
    find_candidates: Callable[[Scene], list[CandidateResult]]
    measure: Callable[[PointCloud, SupportPlane], Cuboid | None]
    estimate_plane: PlaneEstimator | None = None


def _detection(cuboid: Cuboid, plane: SupportPlane, number: int) -> dict:
    center = np.asarray(cuboid.base_center_mm, dtype=float)
    axis = np.asarray(cuboid.length_axis, dtype=float)
    sides = [float(cuboid.length_mm), float(cuboid.width_mm)]
    height = None if cuboid.height_mm is None else float(cuboid.height_mm)
    if height is not None:
        sides.append(height)
    if (
        center.shape != (3,)
        or not (np.abs(center) <= MAX_RESULT_COORDINATE_MM).all()
        or axis.shape != (3,)
        or not np.isfinite(axis).all()
        or abs(axis @ axis - 1.0) > 1e-6
        or abs(axis @ plane.normal) > 1e-6
        or abs(center @ plane.normal + plane.offset_mm) > 0.05
        or any(not 1e-6 <= side <= 100_000 for side in sides)
    ):
        raise ValueError("Invalid component cuboid geometry")
    return {
        "id": f"prediction-{number}",
        "base_center_mm": center.tolist(),
        "length_axis": axis.tolist(),
        "length_mm": sides[0],
        "width_mm": sides[1],
        "height_mm": height,
    }


def failure(snapshot: dict, code: str) -> dict:
    return {
        "schema_version": "boxlab-run-result/1",
        "snapshot_id": snapshot["snapshot_id"],
        "frame": snapshot["cloud_frame"],
        "status": "failed",
        "failure_code": code,
        "plane_estimate": None,
        "detections": [],
    }


def process_scene(
    snapshot_id: str, scene: Scene, components: Components, input_mode: str
) -> dict:
    plane = scene.plane
    if (
        input_mode == "estimated_plane"
        and not abs(plane.offset_mm) <= MAX_RESULT_COORDINATE_MM
    ):
        raise ValueError("Estimated plane offset exceeds Workbench result bounds")

    detections = []
    xyz = scene.cloud.xyz_mm
    candidates = components.find_candidates(scene)
    if not isinstance(candidates, list):
        raise ValueError(
            "Component find_candidates must return a list of CandidateResult objects"
        )
    for candidate in candidates:
        if not isinstance(candidate, CandidateResult) or not isinstance(
            candidate.segment, Segment
        ):
            raise ValueError("Invalid component candidate result")
        indices = candidate.segment.point_indices
        if (
            not isinstance(indices, np.ndarray)
            or indices.ndim != 1
            or indices.dtype != np.int32
            or indices.size == 0
            or indices.min() < 0
            or indices.max() >= len(xyz)
            or (
                not np.all(indices[1:] > indices[:-1])
                and np.unique(indices).size != indices.size
            )
        ):
            raise ValueError("Invalid component segment indices")
        indices = indices.copy()
        indices.flags.writeable = False
        decision = candidate.decision
        if not isinstance(decision, str) or decision not in {
            "eligible",
            "background",
            "partially_visible",
            "uncertain",
        }:
            raise ValueError("Invalid component eligibility decision")
        if decision == "eligible":
            parcel = np.take(xyz, indices, axis=0)
            parcel.flags.writeable = False
            cuboid = components.measure(PointCloud(parcel, scene.cloud.frame), plane)
            if cuboid is not None:
                if len(detections) == 1000:
                    raise ValueError("Too many component cuboids for one snapshot")
                detections.append(_detection(cuboid, plane, len(detections) + 1))

    return {
        "schema_version": "boxlab-run-result/1",
        "snapshot_id": snapshot_id,
        "frame": scene.cloud.frame,
        "status": "success",
        "failure_code": None,
        "plane_estimate": {
            "normal": list(plane.normal),
            "offset_mm": plane.offset_mm,
            "frame": plane.frame,
            "units": "mm",
        }
        if input_mode == "estimated_plane"
        else None,
        "detections": detections,
    }


def load_components(settings):
    from copy import deepcopy
    from importlib import import_module

    module = import_module(
        "depthai_nodes.algorithms.parcel."
        "fast_baseline_algorithm_2f65a6f6296d48e981f5fe3f77bfde89.factory"
    )
    return module, module.create_components(deepcopy(settings.get("configuration", {})))
