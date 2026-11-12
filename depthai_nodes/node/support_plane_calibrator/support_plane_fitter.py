"""Horizontal support plane candidates in an organized millimeter point cloud."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from depthai_nodes.message.support_plane import SupportPlane, SupportPlaneCandidate

from .plane_calibration_config import PlaneCalibrationConfig
from .plane_validation import is_valid_horizontal_plane, orient_along_gravity

if TYPE_CHECKING:
    import open3d as o3d

logger = logging.getLogger(__name__)


class PlaneFit(NamedTuple):
    """One horizontal RANSAC plane.

    Attributes:
        inlier_count: RANSAC inliers on the preprocessed cloud.
        plane_eq: Unit plane equation ``(a, b, c, d)`` oriented along gravity.
        centroid: Mean of the inlier points, in millimeters.
    """

    inlier_count: int
    plane_eq: np.ndarray
    centroid: np.ndarray


class SupportPlaneFitter:
    """Find horizontal support plane candidates in an organized point cloud.

    Args:
        cloud_size: Organized cloud ``(width, height)`` in points.
        config: Plane-fitting settings.
    """

    def __init__(
        self, cloud_size: tuple[int, int], config: PlaneCalibrationConfig
    ) -> None:
        """Create a fitter for clouds of one size.

        Args:
            cloud_size: Organized cloud ``(width, height)`` in points.
            config: Plane-fitting settings.
        """
        self._width, self._height = cloud_size
        self._config = config

    @property
    def cloud_size(self) -> tuple[int, int]:
        """Organized cloud ``(width, height)`` in points."""
        return self._width, self._height

    def find_candidates(
        self, points: np.ndarray, gravity: np.ndarray
    ) -> list[SupportPlaneCandidate]:
        """Find the horizontal planes below the camera.

        Near-parallel planes through the same surface count once, and the kept
        plane reports only its own inliers. Counts are taken on the preprocessed
        cloud, and ratios divide by its size before any plane is removed.

        Args:
            points: ``(width * height, 3)`` cloud points in millimeters; points
                with no depth have ``z`` of zero.
            gravity: Raw accelerometer vector.

        Returns:
            The candidates, empty when no plane qualifies.

        Raises:
            ValueError: If ``points`` does not have the organized shape.
        """
        expected = (self._width * self._height, 3)
        if points.shape != expected:
            raise ValueError(
                f"Point cloud is not organized to {(self._width, self._height)}: "
                f"got shape {points.shape}, expected {expected}"
            )
        cloud = self._preprocess(points)
        if cloud is None:
            return []
        fits = self._fit_planes(cloud, gravity)
        if not fits:
            logger.debug("RANSAC rejected: no horizontal support plane candidates")
            return []
        return self._rank(self._drop_duplicates(fits), len(cloud.points))

    def _preprocess(self, points: np.ndarray) -> o3d.geometry.PointCloud | None:
        """Drop empty points, downsample and denoise; ``None`` if too few remain."""
        # open3d is an optional extra, so import it only when fitting.
        import open3d as o3d

        points = points[points[:, 2] > 1e-6]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud = cloud.voxel_down_sample(voxel_size=self._config.voxel_size_mm)
        cloud, _ = cloud.remove_statistical_outlier(20, 0.1)
        if len(cloud.points) < self._config.ransac_sample_points:
            logger.debug(
                "RANSAC rejected: not enough points after filtering "
                "(got %d, need at least %d)",
                len(cloud.points),
                self._config.ransac_sample_points,
            )
            return None
        return cloud

    def _fit_planes(
        self, cloud: o3d.geometry.PointCloud, gravity: np.ndarray
    ) -> list[PlaneFit]:
        """Extract planes with repeated RANSAC, keeping the horizontal ones."""
        config = self._config
        min_inliers = max(
            config.ransac_sample_points,
            # Round first so float noise (0.07 * 100 = 7.000000000000001) does
            # not push an exact count up by one.
            math.ceil(round(config.min_inlier_ratio * len(cloud.points), 9)),
        )
        fits: list[PlaneFit] = []
        for _ in range(config.max_plane_passes):
            if len(cloud.points) < min_inliers:
                break
            try:
                plane_eq, inliers = cloud.segment_plane(
                    config.distance_threshold_mm,
                    config.ransac_sample_points,
                    config.ransac_max_iterations,
                )
            except RuntimeError as exc:
                logger.debug("RANSAC failed during plane segmentation: %s", exc)
                break
            if len(inliers) < min_inliers:
                logger.debug(
                    "RANSAC rejected: too few inliers (got %d, need at least %d)",
                    len(inliers),
                    min_inliers,
                )
                break
            plane_eq = np.asarray(plane_eq, dtype=np.float32)
            norm = float(np.linalg.norm(plane_eq[:3]))
            if norm < 1e-6:
                break
            plane_eq = orient_along_gravity(plane_eq / norm, gravity)
            if is_valid_horizontal_plane(
                plane_eq, gravity, config.max_gravity_angle_deg
            ):
                centroid = np.asarray(cloud.points)[inliers].mean(axis=0)
                fits.append(PlaneFit(len(inliers), plane_eq, centroid))
            cloud = cloud.select_by_index(inliers, invert=True)
        return fits

    def _drop_duplicates(self, fits: list[PlaneFit]) -> list[PlaneFit]:
        """Keep one plane per surface, the one with the most inliers.

        Noisy depth can make RANSAC split one surface into near-parallel planes.
        """
        unique: list[PlaneFit] = []
        for fit in sorted(fits, key=lambda f: -f.inlier_count):
            if any(
                self._same_surface(
                    fit.plane_eq, fit.centroid, kept.plane_eq, kept.centroid
                )
                for kept in unique
            ):
                logger.debug(
                    "Support plane candidate dropped as duplicate (offset=%.2f)",
                    fit.plane_eq[3],
                )
                continue
            unique.append(fit)
        return unique

    def _same_surface(
        self,
        eq_a: np.ndarray,
        centroid_a: np.ndarray,
        eq_b: np.ndarray,
        centroid_b: np.ndarray,
    ) -> bool:
        """Check whether each plane passes near the other's inlier centroid."""
        dist_ab = abs(float(np.dot(eq_a[:3], centroid_b) + eq_a[3]))
        dist_ba = abs(float(np.dot(eq_b[:3], centroid_a) + eq_b[3]))
        return max(dist_ab, dist_ba) < self._config.merge_distance_threshold_mm

    @staticmethod
    def _rank(fits: list[PlaneFit], cloud_size: int) -> list[SupportPlaneCandidate]:
        """Order by inlier count, then by distance to the camera."""
        ranked = sorted(fits, key=lambda f: (-f.inlier_count, abs(f.plane_eq[3])))
        return [
            SupportPlaneCandidate(
                plane=SupportPlane(
                    normal=(
                        float(fit.plane_eq[0]),
                        float(fit.plane_eq[1]),
                        float(fit.plane_eq[2]),
                    ),
                    offset_mm=float(fit.plane_eq[3]),
                ),
                inlier_count=fit.inlier_count,
                inlier_ratio=fit.inlier_count / cloud_size,
            )
            for fit in ranked
        ]
