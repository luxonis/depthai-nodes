"""The two computation callbacks from the pinned Runner contract."""

from collections.abc import Callable
from dataclasses import dataclass

from .scene import CandidateResult, Cuboid, PointCloud, Scene, SupportPlane


@dataclass(frozen=True)
class Components:
    """Callbacks bound to one configuration with scene-local projection reuse."""

    find_candidates: Callable[[Scene], list[CandidateResult]]
    measure: Callable[[PointCloud, SupportPlane], Cuboid | None]
