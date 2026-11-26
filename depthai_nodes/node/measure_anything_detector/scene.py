"""DepthAI message conversion into the frozen MeasureAnything computation types."""

from __future__ import annotations

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.algorithms.measure_anything import CameraView, PointCloud, Scene
from depthai_nodes.algorithms.measure_anything import SupportPlane as AlgorithmPlane
from depthai_nodes.message.measure_anything import _MeasureAnythingSourceTiming
from depthai_nodes.message.support_plane import SupportPlane


class SceneInputError(ValueError):
    """A source frame is incomplete or cannot satisfy the calibrated scene contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _check_cloud_configuration(pointCloudConfig: dai.PointCloudConfig) -> None:
    """Check the native generator's reference-frame and millimeter contract."""
    if (
        pointCloudConfig.getCoordinateSystemType()
        != dai.PointCloudConfig.CoordinateSystemType.DEFAULT
        or pointCloudConfig.getLengthUnit() != dai.LengthUnit.MILLIMETER
        or pointCloudConfig.getUseSpecTranslation()
        or not np.array_equal(pointCloudConfig.getTransformationMatrix(), np.eye(4))
    ):
        raise ValueError(
            "MeasureAnythingDetector requires DEFAULT point-cloud coordinates, millimeters, "
            "calibrated extrinsics, and an identity custom transform"
        )


def _source_timing(
    streamName: str, message: dai.Buffer
) -> _MeasureAnythingSourceTiming:
    """Snapshot one source message's native frame identity."""
    return _MeasureAnythingSourceTiming(
        stream=streamName,
        timestamp=message.getTimestamp(),
        timestamp_device=message.getTimestampDevice(),
        sequence_num=message.getSequenceNum(),
    )


def _upward_support_plane(supportPlane: SupportPlane) -> SupportPlane:
    """Orient toward the reference camera, which must be above the support plane."""
    if supportPlane.offset_mm == 0:
        raise SceneInputError("plane_through_reference_origin")
    if supportPlane.offset_mm < 0:
        return SupportPlane(
            normal=tuple(-value for value in supportPlane.normal),
            offset_mm=-supportPlane.offset_mm,
        )
    return supportPlane


def _pose_to_reference(metadata: dai.ImgTransformation):
    try:
        if metadata is None or not metadata.isValid():
            raise SceneInputError("missing_camera_metadata")
        extrinsics = metadata.getExtrinsics()
        if not extrinsics.hasValidRotationMatrix():
            raise SceneInputError("missing_camera_metadata")
        pose = np.asarray(
            extrinsics.getTransformationMatrix(False, dai.LengthUnit.MILLIMETER),
            dtype=np.float64,
        )
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-9, rtol=0)
            or np.linalg.det(pose[:3, :3]) == 0
        ):
            raise SceneInputError("invalid_camera_metadata")
        return (extrinsics.toDeviceId, extrinsics.toCameraSocket), pose
    except RuntimeError as exc:
        raise SceneInputError("invalid_camera_metadata") from exc


def prepare_scene(
    pointCloud: dai.PointCloudData,
    color: dai.ImgFrame,
    left: dai.ImgFrame,
    right: dai.ImgFrame,
    supportPlane: SupportPlane,
) -> Scene:
    """Decode finite millimeter XYZ and RGB/mono images on their calibrated grids.

    Color input accepts NV12 and native 8-bit RGB/BGR layouts. Mono views accept
    GRAY8 or RAW8. DepthAI's ``getCvFrame`` handles the delivered row layout and returns
    BGR for color, which is converted to RGB before algorithm invocation.
    Calibration comes from each message's ``ImgTransformation``, including
    crop/scale intrinsics, distortion, and pose in millimeters. XYZ must already
    occupy the common calibration reference frame, as ordinary native DEFAULT
    point clouds with an identity custom transform do. The supplied supportPlane
    uses those same coordinates. The views need not share image grids or be the
    exact inputs used to generate the cloud. They must capture the same scene
    interval and carry calibration for their actual pixels, including any
    rectification, crop, or scale. No image resize or point filtering occurs here.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        xyz = np.array(pointCloud.getPoints(), dtype=np.float32, copy=True)
    if xyz.size == 0:
        xyz = xyz.reshape(0, 3)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise SceneInputError("invalid_point_cloud")
    xyz.flags.writeable = False
    reference, _ = _pose_to_reference(pointCloud.getTransformation())
    frame = f"{reference[0]}/{reference[1].name}"
    views = []
    for role, imgFrame in zip(("color", "left", "right"), (color, left, right)):
        metadata = imgFrame.getTransformation()
        cameraReference, pose = _pose_to_reference(metadata)
        if cameraReference != reference:
            raise SceneInputError("incompatible_camera_frames")
        size = (imgFrame.getWidth(), imgFrame.getHeight())
        if metadata.getSize() != size:
            raise SceneInputError("calibration_grid_mismatch")
        intrinsics = np.asarray(metadata.getIntrinsicMatrix(), dtype=np.float64)
        coefficients = np.asarray(
            metadata.getDistortionCoefficients(), dtype=np.float64
        )
        if (
            intrinsics.shape != (3, 3)
            or not np.isfinite(intrinsics).all()
            or np.linalg.det(intrinsics) == 0
            or coefficients.ndim != 1
            or not np.isfinite(coefficients).all()
        ):
            raise SceneInputError("invalid_camera_metadata")
        model = metadata.getDistortionModel()
        if model == dai.CameraModel.Perspective and len(coefficients) in (
            0,
            4,
            5,
            8,
            12,
            14,
        ):
            distortion = "perspective"
        elif model == dai.CameraModel.Fisheye and len(coefficients) == 4:
            distortion = "fisheye"
        else:
            raise SceneInputError("unsupported_distortion_model")
        referenceToCamera = np.linalg.inv(pose)
        if role == "color":
            if imgFrame.getType() not in (
                dai.ImgFrame.Type.NV12,
                dai.ImgFrame.Type.RGB888i,
                dai.ImgFrame.Type.RGB888p,
                dai.ImgFrame.Type.BGR888i,
                dai.ImgFrame.Type.BGR888p,
            ):
                raise SceneInputError("invalid_color_format")
            pixels = cv2.cvtColor(imgFrame.getCvFrame(), cv2.COLOR_BGR2RGB)
            expected = (*size[::-1], 3)
        else:
            if imgFrame.getType() not in (
                dai.ImgFrame.Type.GRAY8,
                dai.ImgFrame.Type.RAW8,
            ):
                raise SceneInputError("invalid_stereo_format")
            pixels = imgFrame.getCvFrame()
            expected = size[::-1]
        if pixels.shape != expected or pixels.dtype != np.uint8:
            raise SceneInputError("invalid_image_pixels")
        pixels.flags.writeable = False
        views.append(
            CameraView(
                role,
                pixels,
                tuple(map(tuple, intrinsics.tolist())),
                distortion,
                tuple(coefficients.tolist()),
                tuple(map(tuple, referenceToCamera.tolist())),
            )
        )
    return Scene(
        PointCloud(xyz, frame),
        AlgorithmPlane(supportPlane.normal, supportPlane.offset_mm, frame),
        tuple(views),
    )
