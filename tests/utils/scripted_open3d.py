"""A scripted stand-in for ``open3d.geometry.PointCloud`` in plane-fitting tests.

Patch ``open3d.geometry.PointCloud`` with `ScriptedScene.cloud` to make RANSAC
return fixed planes, raise, or run a callback while a fit is in progress.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Plane equation (a, b, c, d) and its inlier count.
ScriptedPlane = Tuple[Sequence[float], int]


class ScriptedScene:
    def __init__(
        self,
        planes: List[ScriptedPlane],
        *,
        error: Optional[Exception] = None,
        during_fit: Optional[Callable[[], None]] = None,
    ):
        self.planes = planes
        self.error = error
        self.during_fit = during_fit

    def cloud(self) -> "ScriptedCloud":
        return ScriptedCloud(self, self.planes)


class ScriptedCloud:
    def __init__(self, scene: ScriptedScene, planes: List[ScriptedPlane]):
        self.scene = scene
        self.planes = planes
        self.points = np.zeros((400, 3))

    def voxel_down_sample(self, voxel_size):
        if self.scene.error is not None:
            raise self.scene.error
        return self

    def remove_statistical_outlier(self, neighbors, ratio):
        return self, list(range(len(self.points)))

    def segment_plane(self, distance, samples, iterations):
        if self.scene.during_fit is not None:
            self.scene.during_fit()
        if not self.planes:
            raise RuntimeError("No plane fit")
        equation, count = self.planes[0]
        return list(equation), list(range(count))

    def select_by_index(self, indices, invert):
        child = ScriptedCloud(self.scene, self.planes[1:])
        child.points = np.asarray(self.points)[len(indices) :]
        return child
