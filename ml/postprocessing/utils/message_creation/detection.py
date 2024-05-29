import depthai as dai
import numpy as np
import cv2

from ....messages.img_detections import (
    ImgDetectionWithKeypoints,
    ImgDetectionsWithKeypoints,
)


def create_detections_msg(
    detections: list,
    include_keypoints: bool = False,
) -> dai.ImgFrame:
    """
    Create a depthai message for an image array.

    @type detections: list
    @ivar detections: List of detections.

    @type include_keypoints: bool
    @ivar include_keypoints: If True, the keypoints are included in the message.
    """

    if include_keypoints:
        img_detection = ImgDetectionWithKeypoints
        img_detections = ImgDetectionsWithKeypoints
    else:
        img_detection = dai.ImgDetection
        img_detections = dai.ImgDetections

    img_detection_list = []
    for detection in detections:
        detections_message = img_detection()
        detections_message.label = detection["label"]
        detections_message.confidence = detection["confidence"]
        detections_message.xmin = detection["xmin"]
        detections_message.ymin = detection["ymin"]
        detections_message.xmax = detection["xmax"]
        detections_message.ymax = detection["ymax"]
        if include_keypoints:
            detections_message.keypoints = detection["keypoints"]
        img_detection_list.append(img_detection)

    detections_message = img_detections()
    detections_message.detections = img_detection_list

    return detections_message
