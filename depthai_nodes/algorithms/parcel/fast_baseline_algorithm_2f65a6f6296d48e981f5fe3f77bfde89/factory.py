"""The frozen geometric candidate with shared projections and native reductions."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from inspect import signature

from .components import Components
from .geometry import detect, measure
from .scene import CandidateResult, Scene
from .visibility import Visibility


def create_components(configuration: dict) -> Components:
    detection = configuration["detection"]
    measurement = configuration["measurement"]
    visibility = Visibility(**configuration["visibility"])
    signature(detect).bind(None, **detection)
    signature(measure).bind(None, None, **measurement)
    parallel = configuration.get("pipeline", {}).get("parallel_validation", False)

    def find_candidates(scene: Scene) -> list[CandidateResult]:
        with ThreadPoolExecutor(max_workers=len(scene.views) or 1) as projection_pool:
            visibility.begin(
                scene, detection["foreground_min_height_mm"], projection_pool
            )
            segments = detect(scene, **detection)
            visibility.prepare(scene, segments)
            if parallel and segments:
                with ThreadPoolExecutor(max_workers=min(4, len(segments))) as pool:
                    decisions = list(
                        pool.map(
                            partial(visibility.validate_candidate, scene), segments
                        )
                    )
            else:
                decisions = [
                    visibility.validate_candidate(scene, segment)
                    for segment in segments
                ]
            return [
                CandidateResult(segment, decision)
                for segment, decision in zip(segments, decisions)
            ]

    return Components(find_candidates, partial(measure, **measurement))
