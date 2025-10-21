from typing import TypeVar, Union

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.img_detections import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
)
from depthai_nodes.message.keypoints import Keypoint, Keypoints
from depthai_nodes.message.segmentation import SegmentationMask

UNASSIGNED_MASK_LABEL = -1


GMessage = TypeVar(
    "GMessage",
    bound=Union[dai.ImgDetections, ImgDetectionsExtended, Keypoints, SegmentationMask],
)


def remap_message(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    message: GMessage,
) -> GMessage:
    if isinstance(message, dai.ImgDetections):
        return remap_img_detections(src_transformation, dst_transformation, message)
    elif isinstance(message, Keypoints):
        return remap_keypoints(src_transformation, dst_transformation, message)
    elif isinstance(message, SegmentationMask):
        return remap_segmentation_mask(src_transformation, dst_transformation, message)
    elif isinstance(message, ImgDetectionsExtended):
        return remap_img_detections_extended(
            src_transformation, dst_transformation, message
        )
    else:
        raise TypeError(f"Unsupported message type: {type(message)}")


def remap_img_detections(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    detections: dai.ImgDetections,
) -> dai.ImgDetections:
    new_detections = dai.ImgDetections()
    new_detections.detections = [
        remap_img_detection(src_transformation, dst_transformation, det)
        for det in detections.detections
    ]
    return new_detections


def remap_img_detections_extended(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    detections: ImgDetectionsExtended,
) -> ImgDetectionsExtended:
    new_detections = ImgDetectionsExtended()
    new_detections.detections = [
        remap_img_detection_extended(src_transformation, dst_transformation, det)
        for det in detections.detections
    ]
    new_detections.masks = remap_segmentation_mask_array(
        src_transformation, dst_transformation, detections.masks
    )
    return new_detections


def remap_segmentation_mask_array(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    segmentation_mask: np.ndarray,
) -> np.ndarray:
    dst_matrix = np.array(dst_transformation.getMatrix())
    src_matrix = np.array(src_transformation.getMatrixInv())
    trans_matrix = dst_matrix @ src_matrix
    new_mask = cv2.warpPerspective(
        segmentation_mask,
        trans_matrix,
        dst_transformation.getSize(),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=UNASSIGNED_MASK_LABEL,  # type: ignore
    )
    return new_mask


def remap_segmentation_mask(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    segmentation_mask: SegmentationMask,
) -> SegmentationMask:
    new_mask = SegmentationMask()
    new_mask.mask = remap_segmentation_mask_array(
        src_transformation, dst_transformation, segmentation_mask.mask
    )
    return new_mask


def remap_img_detection(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    img_detection: dai.ImgDetection,
) -> dai.ImgDetection:
    new_det = dai.ImgDetection()
    min_pt = src_transformation.remapPointTo(
        dst_transformation,
        dai.Point2f(
            np.clip(img_detection.xmin, 0, 1), np.clip(img_detection.ymin, 0, 1)
        ),
    )
    max_pt = src_transformation.remapPointTo(
        dst_transformation,
        dai.Point2f(
            np.clip(img_detection.xmax, 0, 1), np.clip(img_detection.ymax, 0, 1)
        ),
    )
    new_det.xmin = max(0, min(min_pt.x, 1))
    new_det.ymin = max(0, min(min_pt.y, 1))
    new_det.xmax = max(0, min(max_pt.x, 1))
    new_det.ymax = max(0, min(max_pt.y, 1))
    new_det.label = img_detection.label
    new_det.confidence = img_detection.confidence
    return new_det


def remap_img_detection_extended(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    detection: ImgDetectionExtended,
) -> ImgDetectionExtended:
    new_det = ImgDetectionExtended()

    if detection.rotated_rect.angle == 0:
        new_rect = src_transformation.remapRectTo(
            dst_transformation, detection.rotated_rect
        )
        new_det.rotated_rect = (
            new_rect.center.x,
            new_rect.center.y,
            new_rect.size.width,
            new_rect.size.height,
            new_rect.angle,
        )
    else:
        # TODO: This is a temporary fix - DepthAI doesn't handle rotated rects with angle != 0 correctly
        pts = detection.rotated_rect.getPoints()
        pts = [dai.Point2f(np.clip(pt.x, 0, 1), np.clip(pt.y, 0, 1)) for pt in pts]
        remapped_pts = [
            src_transformation.remapPointTo(dst_transformation, pt) for pt in pts
        ]
        remapped_pts = [
            (np.clip(pt.x, 0, 1), np.clip(pt.y, 0, 1)) for pt in remapped_pts
        ]
        (center_x, center_y), (width, height), angle = cv2.minAreaRect(
            np.array(remapped_pts, dtype=np.float32)
        )
        new_det.rotated_rect = (
            center_x,
            center_y,
            width,
            height,
            angle,
        )
    new_det.confidence = detection.confidence
    new_det.label = detection.label
    new_det.label_name = detection.label_name

    new_kpts_list = []
    for kpt in detection.keypoints:
        new_kpt = remap_keypoint(src_transformation, dst_transformation, kpt)
        new_kpts_list.append(new_kpt)
    new_kpts = Keypoints()
    new_kpts.keypoints = new_kpts_list
    new_det.keypoints = new_kpts
    return new_det


def remap_keypoint(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    keypoint: Keypoint,
) -> Keypoint:
    new_kpt = Keypoint()
    new_kpt.x = src_transformation.remapPointTo(
        dst_transformation, dai.Point2f(keypoint.x, keypoint.y)
    ).x
    new_kpt.y = src_transformation.remapPointTo(
        dst_transformation, dai.Point2f(keypoint.x, keypoint.y)
    ).y
    new_kpt.z = keypoint.z
    new_kpt.confidence = keypoint.confidence
    new_kpt.label_name = keypoint.label_name
    return new_kpt


def remap_keypoints(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    keypoints: Keypoints,
) -> Keypoints:
    new_kpts_list = []
    for kpt in keypoints.keypoints:
        new_kpt = remap_keypoint(src_transformation, dst_transformation, kpt)
        new_kpts_list.append(new_kpt)
    new_kpts = Keypoints()
    new_kpts.keypoints = new_kpts_list
    return new_kpts
