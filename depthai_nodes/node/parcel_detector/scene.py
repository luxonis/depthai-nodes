"""DepthAI message conversion into the frozen parcel computation types."""

from __future__ import annotations

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.algorithms.parcel import CameraView, PointCloud, Scene
from depthai_nodes.algorithms.parcel import SupportPlane as AlgorithmPlane
from depthai_nodes.message.parcel import (
    ParcelPlane,
    ParcelRigCalibration,
    ParcelSourceTiming,
)
from depthai_nodes.message.support_plane import SupportPlane


class SceneInputError(ValueError):
    """A source frame is incomplete or cannot satisfy the calibrated scene contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def source_timing(stream: str, message: dai.Buffer) -> ParcelSourceTiming:
    """Snapshot one source message's native frame identity."""
    return ParcelSourceTiming(
        stream=stream,
        timestamp=message.getTimestamp(),
        timestamp_device=message.getTimestampDevice(),
        sequence_num=message.getSequenceNum(),
    )


def upward_plane(
    message: ParcelPlane | None, calibration: ParcelRigCalibration
) -> SupportPlane:
    """Snapshot and orient the supplied plane against the declared scene gravity."""
    if not isinstance(message, ParcelPlane) or message.plane is None:
        raise SceneInputError("missing_support_plane")
    if message.frame != calibration.frame:
        raise SceneInputError("plane_frame_mismatch")
    plane = message.plane
    direction = float(np.dot(plane.normal, calibration.down))
    if direction == 0:
        raise SceneInputError("plane_not_upward")
    if direction > 0:
        return SupportPlane(
            normal=tuple(-v for v in plane.normal), offset_mm=-plane.offset_mm
        )
    return SupportPlane(normal=plane.normal, offset_mm=plane.offset_mm)


def prepare_scene(
    cloud: dai.PointCloudData,
    images: tuple[dai.ImgFrame, dai.ImgFrame, dai.ImgFrame],
    plane: SupportPlane,
    calibration: ParcelRigCalibration,
) -> Scene:
    """Decode finite millimeter XYZ and RGB/mono images on their calibrated grids.

    Color input accepts NV12 and native 8-bit RGB/BGR layouts. Stereo must be
    GRAY8. DepthAI's ``getCvFrame`` handles the delivered row layout and returns
    BGR for color, which is converted to RGB before algorithm invocation.
    No geometric image resize or point filtering occurs here.
    """
    if not isinstance(cloud, dai.PointCloudData):
        raise SceneInputError("invalid_point_cloud")
    with np.errstate(over="ignore", invalid="ignore"):
        xyz = np.array(cloud.getPoints(), dtype=np.float32, copy=True)
    if xyz.size == 0:
        xyz = xyz.reshape(0, 3)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise SceneInputError("invalid_point_cloud")
    xyz.flags.writeable = False
    views = []
    for image, camera in zip(images, calibration.cameras):
        if not isinstance(image, dai.ImgFrame):
            raise SceneInputError("missing_source_image")
        if (image.getWidth(), image.getHeight()) != camera.size:
            raise SceneInputError("calibration_grid_mismatch")
        if camera.role == "color":
            if image.getType() not in (
                dai.ImgFrame.Type.NV12,
                dai.ImgFrame.Type.RGB888i,
                dai.ImgFrame.Type.RGB888p,
                dai.ImgFrame.Type.BGR888i,
                dai.ImgFrame.Type.BGR888p,
            ):
                raise SceneInputError("invalid_color_format")
            pixels = cv2.cvtColor(image.getCvFrame(), cv2.COLOR_BGR2RGB)
            expected = (*camera.size[::-1], 3)
        else:
            if image.getType() != dai.ImgFrame.Type.GRAY8:
                raise SceneInputError("invalid_stereo_format")
            pixels = image.getCvFrame()
            expected = camera.size[::-1]
        if pixels.shape != expected or pixels.dtype != np.uint8:
            raise SceneInputError("invalid_image_pixels")
        pixels.flags.writeable = False
        views.append(
            CameraView(
                camera.role,
                pixels,
                camera.intrinsics,
                camera.distortion_model,
                camera.distortion_coefficients,
                camera.scene_to_camera_mm,
            )
        )
    return Scene(
        PointCloud(xyz, calibration.frame),
        AlgorithmPlane(plane.normal, plane.offset_mm, calibration.frame),
        tuple(views),
    )
