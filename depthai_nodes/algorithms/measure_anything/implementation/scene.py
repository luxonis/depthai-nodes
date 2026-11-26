"""Scene types and calibrated 3D-to-image projection for the measurement algorithm."""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]
Matrix3 = tuple[Vec3, Vec3, Vec3]
Matrix4 = tuple[Vec4, Vec4, Vec4, Vec4]
CameraRole = Literal["color", "left", "right"]
CandidateDecision = Literal["eligible", "background", "partially_visible", "uncertain"]
InputMode = Literal["supplied_plane", "estimated_plane"]


@dataclass(frozen=True)
class PointCloud:
    xyz_mm: NDArray[np.float32]
    frame: str


@dataclass(frozen=True)
class SupportPlane:
    normal: Vec3
    offset_mm: float
    frame: str


@dataclass(frozen=True)
class CameraView:
    role: CameraRole
    image: NDArray[np.uint8]
    intrinsics: Matrix3
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    scene_to_camera_mm: Matrix4


@dataclass(frozen=True)
class Scene:
    cloud: PointCloud
    plane: SupportPlane
    views: tuple[CameraView, ...]


@dataclass(frozen=True)
class Segment:
    point_indices: NDArray[np.int32]


@dataclass(frozen=True)
class CandidateResult:
    segment: Segment
    decision: CandidateDecision


@dataclass(frozen=True)
class Cuboid:
    base_center_mm: Vec3
    length_axis: Vec3
    length_mm: float
    width_mm: float
    height_mm: float | None


PlaneEstimator = Callable[[PointCloud, tuple[CameraView, ...], Vec3], SupportPlane]
CAMERA_ROLES: tuple[CameraRole, ...] = ("color", "left", "right")


class SceneError(ValueError):
    """A frozen artifact or its scene metadata could not be decoded."""


def project_points(xyz_mm: NDArray, view: CameraView) -> NDArray[np.float64]:
    """Return (u, v, camera Z in mm) in input order, with NaN rows for invalid
    projections.

    Finite points outside the image remain available for the caller to clip.
    """
    import cv2

    xyz = np.asarray(xyz_mm, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise SceneError("Projection needs an (N, 3) array")
    transform = np.asarray(view.scene_to_camera_mm)
    with np.errstate(over="ignore", invalid="ignore"):
        camera_xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    valid = (camera_xyz[:, 2] > 0) & np.isfinite(camera_xyz).all(axis=1)
    out = np.full(camera_xyz.shape, np.nan)
    if not valid.any():
        return out

    camera = camera_xyz[valid]
    distortion = np.asarray(view.distortion_coefficients, dtype=np.float64)
    model = view.distortion_model.casefold()
    if model == "perspective" and distortion.size in (4, 5, 8, 12, 14):
        normalized = np.empty((len(camera), 2))
        zero = np.zeros(3)
        identity = np.eye(3)
        # OpenCV allocates a projection Jacobian per call.
        for start in range(0, len(camera), 32768):
            subset = slice(start, start + 32768)
            projected, _ = cv2.projectPoints(
                camera[subset], zero, zero, identity, distortion
            )
            normalized[subset] = projected.reshape(-1, 2)
    elif (model == "perspective" and distortion.size == 0) or (
        model == "none" and not distortion.any()
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = camera[:, :2] / camera[:, 2:]
    elif model == "fisheye" and distortion.size == 4:
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = camera[:, :2] / camera[:, 2:]
        normalized = cv2.fisheye.distortPoints(
            normalized.reshape(-1, 1, 2), np.eye(3), distortion
        ).reshape(-1, 2)
    else:
        raise SceneError(
            f"Unsupported distortion model or coefficient count for {view.role}"
        )

    intrinsics = np.asarray(view.intrinsics)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pixels = normalized @ intrinsics[:, :2].T + intrinsics[:, 2]
        out[valid, :2] = pixels[:, :2] / pixels[:, 2:]
    out[valid, 2] = camera[:, 2]
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out
