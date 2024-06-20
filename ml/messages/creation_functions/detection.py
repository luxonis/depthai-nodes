import numpy as np
from typing import List, Tuple
import depthai as dai

from messages.img_detections import (
    ImgDetectionWithKeypoints,
    ImgDetectionsWithKeypoints,
)


def create_detection_message(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: List[int] = None,
    keypoints: List[List[Tuple[float, float]]] = None,
) -> dai.ImgDetections:
    """
    Create a message for the detection. The message contains the bounding boxes, labels, and confidence scores of detected objects.
    If there are no labels or we only have one class, we can set labels to None and all detections will have label set to 0.

    Args:
        bboxes (np.ndarray): Detected bounding boxes of shape (N,4) meaning [...,[x_min, y_min, x_max, y_max],...].
        scores (np.ndarray): Confidence scores of detected objects of shape (N,).
        labels (List[int], optional): Labels of detected objects of shape (N,). Defaults to None.
        keypoints (List[List[Tuple[float, float]]], optional): Keypoints of detected objects of shape (N,2). Defaults to None.

    Returns:
        dai.ImgDetections OR ImgDetectionsWithKeypoints: Message containing the bounding boxes, labels, confidence scores, and keypoints of detected objects.
    """

    # checks for bboxes
    if not isinstance(bboxes, np.ndarray):
        raise ValueError(f"bboxes should be numpy array, got {type(bboxes)}.")
    if len(bboxes) != 0:
        if len(bboxes.shape) != 2:
            raise ValueError(
                f"bboxes should be of shape (N,4) meaning [...,[x_min, y_min, x_max, y_max],...], got {bboxes.shape}."
            )
        if bboxes.shape[1] != 4:
            raise ValueError(
                f"bboxes 2nd dimension should be of size 4 e.g. [x_min, y_min, x_max, y_max] got {bboxes.shape[1]}."
            )
        x_valid = bboxes[:, 0] < bboxes[:, 2]
        y_valid = bboxes[:, 1] < bboxes[:, 3]
        if not (np.all(x_valid) and np.all(y_valid)):
            raise ValueError(
                f"bboxes should be in format [x_min, y_min, x_max, y_max] where xmin < xmax and ymin < ymax."
            )

    # checks for scores
    if not isinstance(scores, np.ndarray):
        raise ValueError(f"scores should be numpy array, got {type(scores)}.")
    if len(scores) != 0:
        if len(scores.shape) != 1:
            raise ValueError(f"scores should be of shape (N,) meaning, got {scores.shape}.")
        if scores.shape[0] != bboxes.shape[0]:
            raise ValueError(
                f"scores should have same length as bboxes, got {scores.shape[0]} and {bboxes.shape[0]}."
            )

    # checks for labels
    if labels is not None and len(labels) != 0:
        if not isinstance(labels, List):
            raise ValueError(f"labels should be list, got {type(labels)}.")
        for label in labels:
            if not isinstance(label, int):
                raise ValueError(
                    f"labels should be list of integers, got {type(label)}."
                )
        if len(labels) != bboxes.shape[0]:
            raise ValueError(
                f"labels should have same length as bboxes, got {len(labels)} and {bboxes.shape[0]}."
            )

    # checks for keypoints
    if keypoints is not None and len(keypoints) != 0:
        if not isinstance(keypoints, List):
            raise ValueError(f"keypoints should be list, got {type(keypoints)}.")
        for pointcloud in keypoints:
            for point in pointcloud:
                if not isinstance(point, Tuple):
                    raise ValueError(
                        f"keypoint pairs should be list of tuples, got {type(point)}."
                    )
        if len(keypoints) != bboxes.shape[0]:
            raise ValueError(
                f"keypoints should have same length as bboxes, got {len(keypoints)} and {bboxes.shape[0]}."
            )

    if keypoints is not None:
        img_detection = ImgDetectionWithKeypoints
        img_detections = ImgDetectionsWithKeypoints
    else:
        img_detection = dai.ImgDetection
        img_detections = dai.ImgDetections

    detections = []
    for i in range(bboxes.shape[0]):
        detection = img_detection()
        detection.xmin = bboxes[i][0]
        detection.ymin = bboxes[i][1]
        detection.xmax = bboxes[i][2]
        detection.ymax = bboxes[i][3]
        detection.confidence = scores[i]
        if labels is None:
            detection.label = 0
        else:
            detection.label = labels[i]
        if keypoints is not None:
            detection.keypoints = keypoints[i]
        detections.append(detection)

    detections_msg = img_detections()
    detections_msg.detections = detections
    return detections_msg
