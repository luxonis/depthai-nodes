from typing import List, Union

import depthai as dai
import numpy as np

from ...messages import (
    CornerDetections,
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Keypoint,
    Keypoints,
    Line,
    Lines,
)
from .keypoints import create_keypoints_message


def create_detection_message(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray = None,
    keypoints: np.ndarray = None,
    keypoints_scores: np.ndarray = None,
    masks: np.ndarray = None,
) -> Union[dai.ImgDetections, ImgDetectionsExtended]:
    """Create a DepthAI message for object detection.

    @param bbox: Bounding boxes of the detected objects of shape (N,4) meaning [...,[x_min, y_min, x_max, y_max],...].
    @type bbox: np.ndarray
    @param scores: Confidence scores of the detected objects of shape (N,).
    @type scores: np.ndarray
    @param labels: Labels of the detected objects of shape (N,).
    @type labels: np.ndarray
    @param keypoints: Keypoints of the detected objects of shape (N,K,2) for 3D keypoints and (N,K,3) for 3D keypoints. K is the number of keypoints per object.
    @type keypoints: np.ndarray
    @param bbox_scores: Confidence scores of the detected keypoints of shape (N,K).
    @type scores: np.ndarray
    @param masks: Masks of the detected objects of shape (N,H,W).
    @type masks: np.ndarray

    @return: Message containing the bounding boxes, confidence scores, and optionally also labels, keypoints, and masks of the detected objects.
    @rtype: Union[dai.ImgDetections, ImgDetectionsExtended]

    @raise ValueError: If bboxes are not a numpy array.
    @raise ValueError: If bboxes are not of shape (N,4).
    @raise ValueError: If bboxes 2nd dimension is not of size 4.
    @raise ValueError: If bboxes are not in format [x_min, y_min, x_max, y_max] where xmin < xmax and ymin < ymax.
    @raise ValueError: If scores are not a numpy array.
    @raise ValueError: If scores are not of shape (N,).
    @raise ValueError: If scores do not have the same length as bboxes.
    @raise ValueError: If labels are not a numpy array.
    @raise ValueError: If labels are not of shape (N,).
    @raise ValueError: If labels do not have the same length as bboxes.
    @raise ValueError: If keypoints are not a numpy array.
    @raise ValueError: If keypoints are not of shape (N,K,2) or (N,K,3).
    @raise ValueError: If keypoints do not have the same length as bboxes.
    @raise ValueError: If keypoints 3nd dimension is not of size 2 or 3.
    @raise ValueError: If keypoint_scores are not a numpy array.
    @raise ValueError: If keypoint_scores are not of shape (N,K).
    @raise ValueError: If keypoint_scores do not have the same length as bboxes
    @raise ValueError: If masks are not a numpy array.
    @raise ValueError: If masks are not of shape (N,H,W).
    @raise ValueError: If masks do not have the same length as bboxes.
    """

    # checks for bboxes
    if not isinstance(bboxes, np.ndarray):
        raise ValueError(f"Bounding boxes should be numpy array, got {type(bboxes)}.")
    if len(bboxes) != 0:
        if len(bboxes.shape) != 2:
            raise ValueError(
                f"Bounding boxes should be of shape (N,4) meaning [...,[x_min, y_min, x_max, y_max],...], got {bboxes.shape}."
            )
        if bboxes.shape[1] != 4:
            raise ValueError(
                f"Bounding boxes 2nd dimension should be of size 4 e.g. [x_min, y_min, x_max, y_max] got {bboxes.shape[1]}."
            )
        x_valid = bboxes[:, 0] < bboxes[:, 2]
        y_valid = bboxes[:, 1] < bboxes[:, 3]
        if not (np.all(x_valid) and np.all(y_valid)):
            raise ValueError(
                "Bounding boxes should be in format [x_min, y_min, x_max, y_max] where xmin < xmax and ymin < ymax."
            )

    number_of_detections = bboxes.shape[0]

    # checks for scores
    if not isinstance(scores, np.ndarray):
        raise ValueError(f"Scores should be numpy array, got {type(scores)}.")
    if len(scores) != 0:
        if len(scores.shape) != 1:
            raise ValueError(f"Scores should be of shape (N,), got {scores.shape}.")
        if scores.shape[0] != number_of_detections:
            raise ValueError(
                f"Scores should have length N. Got {scores.shape[0]}, expected {number_of_detections}."
            )

    # checks for labels
    if labels is not None and len(labels) != 0:
        if not isinstance(labels, np.ndarray):
            raise ValueError(f"Labels should be numpy array, got {type(labels)}.")
        if len(labels.shape) != 1:
            raise ValueError(f"Labels should be of shape (N,), got {labels.shape}.")
        if len(labels) != number_of_detections:
            raise ValueError(
                f"Labels should have length N. Got {len(labels)}, expected {number_of_detections}."
            )

    # checks for keypoints
    if keypoints is not None and len(keypoints) != 0:
        if not isinstance(keypoints, np.ndarray):
            raise ValueError(f"Keypoints should be numpy array, got {type(keypoints)}.")
        if len(keypoints.shape) != 3:
            raise ValueError(
                f"Keypoints should be of shape (N,K,2) or (N,K,3), got {labels.shape}."
            )
        if len(keypoints) != number_of_detections:
            raise ValueError(
                f"Keypoints should have length N. Got {len(keypoints)}, expected {number_of_detections}."
            )
        if keypoints.shape[2] not in [2, 3]:
            raise ValueError(
                f"Keypoint should consist of 2 (x,y) or 3 (x,y,z) coordinates, got {keypoints.shape[2]}."
            )

    # checks for keypoint scores
    if keypoints_scores is not None and len(keypoints_scores) != 0:
        if not isinstance(keypoints_scores, np.ndarray):
            raise ValueError(
                f"Keypoint scores should be numpy array, got {type(keypoints_scores)}."
            )
        if len(keypoints_scores) != 0:
            if len(keypoints_scores.shape) != 2:
                raise ValueError(
                    f"Keypoints Scores should be of shape (N,K), got {keypoints_scores.shape}."
                )
            if len(keypoints_scores) != number_of_detections:
                raise ValueError(
                    f"Keypoints Scores should have length N. Got {len(keypoints_scores)}, expected {number_of_detections}."
                )

    # checks for masks
    if masks is not None and len(masks) != 0:
        if not isinstance(masks, np.ndarray):
            raise ValueError(f"Masks should be numpy array, got {type(masks)}.")
        if len(masks.shape) != 3:
            raise ValueError(
                f"Keypoints should be of shape (N,H,W), got {labels.shape}."
            )
        if len(masks) != number_of_detections:
            raise ValueError(
                f"Masks should have length N. Got {len(masks)}, expected {number_of_detections}."
            )

    if keypoints is not None or masks is not None:
        img_detection = ImgDetectionExtended
        img_detections = ImgDetectionsExtended
    else:
        img_detection = dai.ImgDetection
        img_detections = dai.ImgDetections

    dets_list = []
    for i in range(number_of_detections):
        det = img_detection()

        # set bbox coordinates
        det.xmin = bboxes[i][0].item()
        det.ymin = bboxes[i][1].item()
        det.xmax = bboxes[i][2].item()
        det.ymax = bboxes[i][3].item()

        # set confidence score
        det.confidence = scores[i].item()

        # set label
        det.label = labels[i].item() if labels is not None else 0

        # set keypoints
        if keypoints is not None:
            kps_list = []
            for ii, keypoint in enumerate(keypoints[i]):
                kp = Keypoint()
                kp.x = keypoint[0].item()
                kp.y = keypoint[1].item()
                if len(keypoint) > 2:
                    kp.z = keypoint[2].item()
                # set keypoint score
                if keypoints_scores is not None:
                    kp.confidence = keypoints_scores[i, ii].item()
                kps_list.append(kp)
            kps = Keypoints()
            kps.keypoints = kps_list
            det.keypoints = kps

        # set mask
        if masks is not None:
            det.mask = masks[i]

        dets_list.append(det)

    detections_msg = img_detections()
    detections_msg.detections = dets_list
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
        raise ValueError(f"Lines should be numpy array, got {type(lines)}.")
    if len(lines) != 0:
        if len(lines.shape) != 2:
            raise ValueError(
                f"Lines should be of shape (N,4) meaning [...,[x_start, y_start, x_end, y_end],...], got {lines.shape}."
            )
        if lines.shape[1] != 4:
            raise ValueError(
                f"Lines 2nd dimension should be of size 4 e.g. [x_start, y_start, x_end, y_end] got {lines.shape[1]}."
            )

    # checks for scores
    if not isinstance(scores, np.ndarray):
        raise ValueError(f"Scores should be numpy array, got {type(scores)}.")

    if len(scores) != 0:
        if len(scores.shape) != 1:
            raise ValueError(
                f"Scores should be of shape (N,) meaning, got {scores.shape}."
            )

        for score in scores:
            if not isinstance(score, (float, np.floating)):
                raise ValueError(f"Scores should be of type float, got {type(score)}.")

    if scores.shape[0] != lines.shape[0]:
        raise ValueError(
            f"Scores should have same length as lines, got {scores.shape[0]} and {lines.shape[0]}."
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


def create_corner_detection_message(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: List[int] = None,
) -> CornerDetections:
    """Create a DepthAI message for an object detection.

    @param bbox: Bounding boxes of detected objects in corner format of shape (N,4,2) meaning [...,[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],...].
    @type bbox: np.ndarray
    @param scores: Confidence scores of detected objects of shape (N,).
    @type scores: np.ndarray
    @param labels: Labels of detected objects of shape (N,).
    @type labels: List[int]

    @return: CornerDetections message containing a list of corners, a list of labels, and a list of scores.
    @rtype: CornerDetections
    """
    if bboxes.shape[0] == 0:
        return CornerDetections()

    if bboxes.shape[1] != 4 or bboxes.shape[2] != 2:
        raise ValueError(
            f"Bounding boxes should be of shape (N,4,2), got {bboxes.shape}."
        )

    if bboxes.shape[0] != len(scores):
        raise ValueError(
            f"Number of bounding boxes and scores should have the same length, got {len(scores)} scores and {bboxes.shape[0]} bounding boxes."
        )

    if labels is not None:
        if len(labels) != len(scores):
            raise ValueError(
                f"Number of labels and scores should have the same length, got {len(labels)} labels and {len(scores)} scores."
            )

    corner_boxes = []

    for bbox in bboxes:
        corner_box = create_keypoints_message(bbox)
        corner_boxes.append(corner_box)

    message = CornerDetections()
    if labels is not None:
        message.labels = labels

    message.detections = corner_boxes
    message.scores = list(scores)

    return message
