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
    """XYZ samples in one metric scene coordinate frame.

    The container retains the supplied array without copying or validating it.
    Computation expects finite float32 points and treats the array as read-only.

    Attributes:
        xyz_mm: An ``(N, 3)`` array in millimeters. Row order defines segment indices.
        frame: Coordinate-frame identifier shared with the support plane.
    """

    xyz_mm: NDArray[np.float32]
    frame: str


@dataclass(frozen=True)
class SupportPlane:
    """A unit-normal plane defining zero height in the scene frame.

    Signed height in millimeters is ``xyz_mm @ normal + offset_mm``. The normal
    points upward from the support surface. The message adapter establishes this
    orientation before constructing the computation scene.

    Attributes:
        normal: Unit normal expressed in the scene coordinate frame.
        offset_mm: Constant term in the plane equation, in millimeters.
        frame: Coordinate-frame identifier shared with the point cloud.
    """

    normal: Vec3
    offset_mm: float
    frame: str


@dataclass(frozen=True)
class CameraView:
    """One delivered image and its calibrated scene-to-camera projection.

    Calibration describes the image's actual pixel grid, including any crop,
    scale or rectification. Transform translations use millimeters.

    Attributes:
        role: The color, left or right view identifier.
        image: Read-only uint8 RGB ``(H, W, 3)`` or mono ``(H, W)`` pixels.
        intrinsics: Camera intrinsic matrix for the delivered image coordinates.
        distortion_model: Perspective, fisheye or undistorted projection model.
        distortion_coefficients: Coefficients for the selected distortion model.
        scene_to_camera_mm: Homogeneous transform from scene XYZ to camera XYZ.
    """

    role: CameraRole
    image: NDArray[np.uint8]
    intrinsics: Matrix3
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    scene_to_camera_mm: Matrix4


@dataclass(frozen=True)
class Scene:
    """Point cloud, support plane and calibrated images for one capture interval.

    The cloud and plane share a coordinate frame. Each view maps that frame to
    its camera coordinates. Scene arrays remain unchanged during detection,
    visibility checks and measurement.

    Attributes:
        cloud: Source XYZ samples whose row ordering defines all segments.
        plane: Upward support plane used for foreground heights and cuboid placement.
        views: Retained calibrated color and mono views for visibility checks.
    """

    cloud: PointCloud
    plane: SupportPlane
    views: tuple[CameraView, ...]


@dataclass(frozen=True)
class Segment:
    """Rows of a scene cloud belonging to one proposed object.

    Attributes:
        point_indices: Unique int32 indices into the scene cloud. Their order is
            retained when object points are extracted.
    """

    point_indices: NDArray[np.int32]


@dataclass(frozen=True)
class CandidateResult:
    """A proposed segment and its measurement-eligibility decision.

    Attributes:
        segment: Source-cloud rows retained for this proposal.
        decision: Eligibility outcome. Only eligible segments enter measurement.
    """

    segment: Segment
    decision: CandidateDecision


@dataclass(frozen=True)
class Cuboid:
    """An oriented footprint and optional height relative to a support plane.

    The base center lies on the plane. The length axis is unit length and tangent
    to the plane. The width axis is the plane normal crossed with the length
    axis. Length is at least width. This container retains geometry without
    validating it; the message adapter checks the emitted result.

    Attributes:
        base_center_mm: Footprint center in the scene frame, in millimeters.
        length_axis: Unit vector along the longer footprint side.
        length_mm: Longer footprint extent in millimeters.
        width_mm: Shorter footprint extent in millimeters.
        height_mm: Supported height above the plane, or ``None`` if unavailable.
    """

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
    """Project scene XYZ into a calibrated image while preserving point order.

    The view's transform and lens model produce image coordinates without
    clipping, resizing or rectifying the image. Points with nonpositive camera Z
    or nonfinite projections produce NaN rows. Finite pixels outside the image
    remain available for the caller to clip.

    Args:
        xyz_mm: An ``(N, 3)`` array of scene coordinates in millimeters.
        view: Image-grid calibration and scene-to-camera transform.

    Returns:
        A float64 ``(N, 3)`` array containing pixel u, pixel v and camera Z in
        millimeters for each input row.

    Raises:
        SceneError: If XYZ has the wrong shape or the lens model and coefficient
            count are unsupported.
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
