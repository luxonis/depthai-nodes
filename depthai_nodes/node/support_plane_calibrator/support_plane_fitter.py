"""Horizontal plane candidates in an organized millimeter point cloud."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from depthai_nodes.message.support_plane import SupportPlane
from depthai_nodes.message.support_plane_calibration import PlaneCalibrationConfig

from .plane_validation import is_valid_horizontal_plane

if TYPE_CHECKING:
    import open3d as o3d

logger = logging.getLogger(__name__)

# Inlier count, unit plane equation and inlier centroid of one RANSAC plane.
PlaneFit = Tuple[int, np.ndarray, np.ndarray]


class SupportPlaneFitter:
    """Find horizontal plane candidates in an organized point cloud.

    Args:
        cloud_size: Organized cloud ``(width, height)`` in points.
        config: Plane-fitting settings.
    """

    def __init__(
        self, cloud_size: Tuple[int, int], config: PlaneCalibrationConfig
    ) -> None:
        """Create a fitter for clouds of one size.

        Args:
            cloud_size: Organized cloud ``(width, height)`` in points.
            config: Plane-fitting settings.
        """
        self._width, self._height = cloud_size
        self._config = config

    def find_candidates(
        self, points: np.ndarray, gravity: np.ndarray
    ) -> List[SupportPlane]:
        """Find planes facing against gravity, best first.

        Planes are ranked by inlier count, then by distance to the camera.
        Near-parallel planes through the same surface count once.

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
            logger.debug("RANSAC rejected: no horizontal plane candidates")
            return []
        return self._rank(self._drop_duplicates(fits))

    def _preprocess(self, points: np.ndarray) -> Optional[o3d.geometry.PointCloud]:
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
    ) -> List[PlaneFit]:
        """Extract planes with repeated RANSAC, keeping the horizontal ones."""
        config = self._config
        min_inliers = max(
            config.ransac_sample_points,
            int(config.min_inlier_ratio * len(cloud.points)),
        )
        fits: List[PlaneFit] = []
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
            plane_eq /= norm
            if is_valid_horizontal_plane(
                plane_eq, gravity, config.max_gravity_angle_deg
            ):
                centroid = np.asarray(cloud.points)[inliers].mean(axis=0)
                fits.append((len(inliers), plane_eq, centroid))
            cloud = cloud.select_by_index(inliers, invert=True)
        return fits

    def _drop_duplicates(self, fits: List[PlaneFit]) -> List[PlaneFit]:
        """Keep one plane per surface, the one with the most inliers.

        Noisy depth can make RANSAC split one surface into near-parallel planes.
        """
        unique: List[PlaneFit] = []
        for fit in sorted(fits, key=lambda f: -f[0]):
            _, plane_eq, centroid = fit
            if any(
                self._same_surface(plane_eq, centroid, kept_eq, kept_centroid)
                for _, kept_eq, kept_centroid in unique
            ):
                logger.debug(
                    "Plane candidate dropped as duplicate (offset=%.2f)", plane_eq[3]
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
    def _rank(fits: List[PlaneFit]) -> List[SupportPlane]:
        """Order by inlier count, then by distance to the camera."""
        ranked = sorted(fits, key=lambda f: (-f[0], abs(f[1][3])))
        return [
            SupportPlane(
                normal=(float(eq[0]), float(eq[1]), float(eq[2])),
                offset_mm=float(eq[3]),
            )
            for _, eq, _ in ranked
        ]
