"""Fast-baseline rules with coarse-to-fine point sampling and NumPy reductions."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from inspect import signature

import numpy as np

from .components import Components
from .geometry import detect, measure
from .numpy_ops import threshold_heights
from .scene import CandidateResult, PointCloud, Scene, Segment
from .visibility import Visibility


def create_components(configuration: dict) -> Components:
    detection = configuration["detection"]
    measurement = configuration["measurement"]
    visibility = Visibility(**configuration["visibility"])
    signature(detect).bind(None, **detection)
    signature(measure).bind(None, None, **measurement)
    pipeline = configuration.get("pipeline", {})
    parallel = pipeline.get("parallel_validation", False)
    workers = pipeline.get("workers", 4)
    if type(workers) is not int or not 1 <= workers <= 8:  # noqa: E721 - preserve the frozen exact-type check
        raise ValueError("workers must be an integer from 1 to 8")
    point_stride = pipeline.get("point_stride", 1)
    fallback_stride = pipeline.get("fallback_point_stride", point_stride)
    sampling_pattern = pipeline.get("sampling_pattern", "stride")
    min_scene_points = pipeline.get("min_scene_points", 16384)
    if type(min_scene_points) is not int or min_scene_points < 1:  # noqa: E721 - preserve the frozen exact-type check
        raise ValueError("min_scene_points must be a positive integer")
    if sampling_pattern not in ("stride", "stratified"):
        raise ValueError("sampling_pattern must be stride or stratified")
    for name, value in (
        ("point_stride", point_stride),
        ("fallback_point_stride", fallback_stride),
    ):
        if type(value) is not int or value < 1:  # noqa: E721 - preserve the frozen exact-type check
            raise ValueError(f"{name} must be a positive integer")
    if fallback_stride > point_stride:
        raise ValueError("fallback_point_stride must not exceed point_stride")
    if (
        type(visibility.projection_stride) is not int  # noqa: E721 - frozen exact-type check
        or visibility.projection_stride < 1
    ):
        raise ValueError("projection_stride must be a positive integer")

    def find_at_stride(scene: Scene, stride: int) -> list[CandidateResult]:
        rows = None
        if stride != 1:
            if sampling_pattern == "stride":
                points = scene.cloud.xyz_mm[::stride]
            else:
                rows = np.arange(0, len(scene.cloud.xyz_mm), stride, dtype=np.int32)
                blocks = np.arange(len(rows), dtype=np.uint64)
                mixed = blocks * np.uint64(2654435761)
                mixed ^= mixed >> np.uint64(16)
                rows = np.minimum(
                    rows.astype(np.int64)
                    + (mixed % np.uint64(stride)).astype(np.int64),
                    len(scene.cloud.xyz_mm) - 1,
                ).astype(np.int32)
                points = scene.cloud.xyz_mm[rows]
                points.flags.writeable = False
            scene = Scene(
                PointCloud(points, scene.cloud.frame), scene.plane, scene.views
            )
        thresholds = (
            detection["foreground_min_height_mm"],
            detection["foreground_max_height_mm"],
            visibility.occlusion_min_height_mm,
        )
        with ThreadPoolExecutor(max_workers=workers) as projection_pool:
            heights = threshold_heights(scene, thresholds)
            visibility.begin(
                scene,
                detection["foreground_min_height_mm"],
                projection_pool,
                heights=heights,
            )
            segments = detect(scene, heights=heights, **detection)
            visibility.prepare(scene, segments, pool=projection_pool)
            if parallel and len(segments) > 1:
                decisions = list(
                    projection_pool.map(
                        partial(visibility.validate_candidate, scene), segments
                    )
                )
            else:
                decisions = [
                    visibility.validate_candidate(scene, segment)
                    for segment in segments
                ]
        return [
            CandidateResult(
                Segment(
                    rows[segment.point_indices]
                    if rows is not None
                    else segment.point_indices * stride
                )
                if stride != 1
                else segment,
                decision,
            )
            for segment, decision in zip(segments, decisions)
        ]

    trace = {"last_strides": ()}

    def find_candidates(scene: Scene) -> list[CandidateResult]:
        if len(scene.cloud.xyz_mm) > np.iinfo(np.int32).max + 1:
            raise ValueError("Cloud has too many rows for int32 segment indices")
        limit = max(1, len(scene.cloud.xyz_mm) // min_scene_points)
        initial = min(point_stride, limit)
        fallback = min(fallback_stride, limit)
        candidates = find_at_stride(scene, initial)
        passes = [initial]
        if fallback != initial and not any(
            c.decision == "eligible" for c in candidates
        ):
            candidates = find_at_stride(scene, fallback)
            passes.append(fallback)
        trace["last_strides"] = tuple(passes)
        return candidates

    find_candidates.trace = trace
    return Components(find_candidates, partial(measure, **measurement))
