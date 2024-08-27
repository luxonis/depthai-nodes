from typing import List, Tuple

import depthai as dai
import numpy as np

from ...messages import (
    ImgDetectionsWithAdditionalOutput,
    ImgDetectionWithAdditionalOutput,
    Line,
    Lines,
)


def create_detection_message(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: List[int] = None,
    keypoints: List[List[Tuple[float, float]]] = None,
    masks: List[np.ndarray] = None,
) -> dai.ImgDetections:
    """Create a DepthAI message for an object detection.

    @param bbox: Bounding boxes of detected objects of shape (N,4) meaning [...,[x_min, y_min, x_max, y_max],...].
    @type bbox: np.ndarray
    @param scores: Confidence scores of detected objects of shape (N,).
    @type scores: np.ndarray
    @param labels: Labels of detected objects of shape (N,).
    @type labels: List[int]
    @param keypoints: Keypoints of detected objects of shape (N,2).
    @type keypoints: Optional[List[List[Tuple[float, float]]]]
    @param masks: Masks of detected objects of shape (N, H/4, W/4).

    @return: Message containing the bounding boxes, labels, confidence scores, and keypoints of detected objects.
    @rtype: dai.ImgDetections OR ImgDetectionsWithAdditionalOutput

    @raise ValueError: If the bboxes are not a numpy array.
    @raise ValueError: If the bboxes are not of shape (N,4).
    @raise ValueError: If the bboxes 2nd dimension is not of size 4.
    @raise ValueError: If the bboxes are not in format [x_min, y_min, x_max, y_max] where xmin < xmax and ymin < ymax.
    @raise ValueError: If the scores are not a numpy array.
    @raise ValueError: If the scores are not of shape (N,).
    @raise ValueError: If the scores do not have the same length as bboxes.
    @raise ValueError: If the labels are not a list.
    @raise ValueError: If each label is not an integer.
    @raise ValueError: If the labels do not have the same length as bboxes.
    @raise ValueError: If the keypoints are not a list.
    @raise ValueError: If each keypoint pair is not a tuple of two floats.
    @raise ValueError: If the keypoints do not have the same length as bboxes.
    @raise ValueError: If the masks are not a list.
    @raise ValueError: If each mask is not a 2D numpy array.
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
                "bboxes should be in format [x_min, y_min, x_max, y_max] where xmin < xmax and ymin < ymax."
            )

    # checks for scores
    if not isinstance(scores, np.ndarray):
        raise ValueError(f"scores should be numpy array, got {type(scores)}.")
    if len(scores) != 0:
        if len(scores.shape) != 1:
            raise ValueError(
                f"scores should be of shape (N,) meaning, got {scores.shape}."
            )
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
        for object_keypoints in keypoints:
            for point in object_keypoints:
                if not isinstance(point, Tuple) and not isinstance(point, List):
                    raise ValueError(
                        f"keypoint pairs should be list of tuples, got {type(point)}."
                    )
        if len(keypoints) != bboxes.shape[0]:
            raise ValueError(
                f"keypoints should have same length as bboxes, got {len(keypoints)} and {bboxes.shape[0]}."
            )
    
    if masks is not None and len(masks) != 0:
        if not isinstance(masks, List):
            raise ValueError(f"masks should be list, got {type(masks)}.")
        for mask in masks:
            if not isinstance(mask, np.ndarray):
                raise ValueError(f"mask should be numpy array, got {type(mask)}.")
            if len(mask.shape) != 2:
                raise ValueError(f"mask should be of shape (H/4, W/4), got {mask.shape}.")

        if len(masks) != bboxes.shape[0]:
            raise ValueError(
                f"masks should have same length as bboxes, got {len(masks)} and {bboxes.shape[0]}."
            )

    if keypoints is not None or masks is not None:
        img_detection = ImgDetectionWithAdditionalOutput
        img_detections = ImgDetectionsWithAdditionalOutput
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
        if masks is not None:
            detection.mask = masks[i]
        detections.append(detection)

    detections_msg = img_detections()
    detections_msg.detections = detections
    return detections_msg


def create_line_detection_message(lines: np.ndarray, scores: np.ndarray):
    """Create a DepthAI message for a line detection.

    @param lines: Detected lines of shape (N,4) meaning [...,[x_start, y_start, x_end, y_end],...].
    @type lines: np.ndarray
    @param scores: Confidence scores of detected lines of shape (N,).
    @type scores: np.ndarray

    @return: Message containing the lines and confidence scores of detected lines.
    @rtype: Lines

    @raise ValueError: If the lines are not a numpy array.
    @raise ValueError: If the lines are not of shape (N,4).
    @raise ValueError: If the lines 2nd dimension is not of size E{4}.
    @raise ValueError: If the scores are not a numpy array.
    @raise ValueError: If the scores are not of shape (N,).
    @raise ValueError: If the scores do not have the same length as lines.
    """

    # checks for lines
    if not isinstance(lines, np.ndarray):
        raise ValueError(f"lines should be numpy array, got {type(lines)}.")
    if len(lines) != 0:
        if len(lines.shape) != 2:
            raise ValueError(
                f"lines should be of shape (N,4) meaning [...,[x_start, y_start, x_end, y_end],...], got {lines.shape}."
            )
        if lines.shape[1] != 4:
            raise ValueError(
                f"lines 2nd dimension should be of size 4 e.g. [x_start, y_start, x_end, y_end] got {lines.shape[1]}."
            )

    # checks for scores
    if not isinstance(scores, np.ndarray):
        raise ValueError(f"scores should be numpy array, got {type(scores)}.")
    if len(scores) != 0:
        if len(scores.shape) != 1:
            raise ValueError(
                f"scores should be of shape (N,) meaning, got {scores.shape}."
            )
        if scores.shape[0] != lines.shape[0]:
            raise ValueError(
                f"scores should have same length as lines, got {scores.shape[0]} and {lines.shape[0]}."
            )

    line_detections = []
    for i, line in enumerate(lines):
        line_detection = Line()
        line_detection.start_point = dai.Point2f(line[0], line[1])
        line_detection.end_point = dai.Point2f(line[2], line[3])
        line_detection.confidence = float(scores[i])
        line_detections.append(line_detection)

    lines_msg = Lines()
    lines_msg.lines = line_detections
    return lines_msg
