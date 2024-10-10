import depthai as dai
import numpy as np

from ...messages import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Line,
    Lines,
)
from ...parsers.utils import transform_to_keypoints


def create_detection_message(
    bboxes: np.ndarray,
    scores: np.ndarray,
    angles: np.ndarray = None,
    labels: np.ndarray = None,
    keypoints: np.ndarray = None,
    masks: np.ndarray = None,
) -> ImgDetectionsExtended:
    """Create a DepthAI message for object detection. The message contains the bounding
    boxes in X_center, Y_center, Width, Height format with optional angles, labels and
    detected object keypoints and masks.

    @param bbox: Bounding boxes of detected objects in (x_center, y_center, width,
        height) format.
    @type bbox: np.ndarray
    @param scores: Confidence scores of the detected objects of shape (N,).
    @type scores: np.ndarray
    @param angles: Angles of detected objects expressed in degrees.
    @type angles: Optional[np.ndarray]
    @param labels: Labels of detected objects of shape (N,).
    @type labels: Optional[np.ndarray]
    @param keypoints: Keypoints of detected objects of shape (N,2) or (N,3).
    @type keypoints: Optional[np.array]
    @param masks: Masks of detected objects of shape (N, H, W).
    @type masks: Optional[np.ndarray]
    @return: Message containing the bounding boxes, labels, confidence scores, and
        keypoints of detected objects.
    @rtype: ImgDetectionsExtended
    @raise ValueError: If the bboxes are not a numpy array.
    @raise ValueError: If the bboxes are not of shape (N,4).
    @raise ValueError: If the scores are not a numpy array.
    @raise ValueError: If the scores are not of shape (N,).
    @raise ValueError: If the scores do not have the same length as bboxes.
    @raise ValueError: If the angles do not have the same length as bboxes.
    @raise ValueError: If the angles are not between -360 and 360.
    @raise ValueError: If the labels are not a list of integers.
    @raise ValueError: If the labels do not have the same length as bboxes.
    @raise ValueError: If the keypoints are not a numpy array of shape (N, M, 2 or 3).
    @raise ValueError: If the masks are not a 3D numpy array of shape (img_height,
        img_width, N) or (N, img_height, img_width).
    @raise ValueError: If the masks are not in the range [0, 1].
    """

    if not isinstance(bboxes, np.ndarray):
        raise ValueError(f"Bboxes should be a numpy array, got {type(bboxes)}.")

    if len(bboxes) == 0:
        return ImgDetectionsExtended()

    if len(bboxes.shape) != 2:
        raise ValueError(
            f"Bounding boxes should be a 2D array like [... , [x_center, y_center, width, height], ... ], got {bboxes.shape}."
        )
    if bboxes.shape[1] != 4:
        raise ValueError(
            f"Bounding boxes should be of shape (n, 4), got {bboxes.shape}."
        )

    if not isinstance(scores, np.ndarray):
        raise ValueError(f"Scores should be a numpy array, got {type(scores)}.")

    if len(scores.shape) != 1:
        raise ValueError(f"Scores should be of shape (N,), got {scores.shape}.")

    n_bboxes = len(bboxes)
    if len(scores) != n_bboxes:
        raise ValueError(
            f"Scores should have same length as bboxes, got {len(scores)} scores and {n_bboxes} bounding boxes."
        )

    if labels is not None:
        if not isinstance(labels, np.ndarray):
            raise ValueError(f"Labels should be a numpy array, got {type(labels)}.")
        if labels.shape[0] != n_bboxes:
            raise ValueError(
                f"Labels should have same length as bboxes, got {len(labels)} and {n_bboxes}."
            )

    if angles is not None:
        if not isinstance(angles, np.ndarray):
            raise ValueError(f"Angles should be a numpy array, got {type(angles)}.")
        if len(angles) != n_bboxes:
            raise ValueError(
                f"Angles should have same length as bboxes, got {len(angles)} and {n_bboxes}."
            )
        if not all(-360 <= angle <= 360 for angle in angles):
            raise ValueError(f"Angles should be between -360 and 360, got {angles}.")

    if keypoints is not None:
        if not isinstance(keypoints, np.ndarray):
            raise ValueError(
                f"Keypoints should be a numpy array, got {type(keypoints)}."
            )
        if len(keypoints.shape) != 3:
            raise ValueError(
                f"Keypoints should be of shape (N, M, 2 or 3) meaning [..., [x, y] or [x, y, z], ...], got {keypoints.shape}."
            )
        n_detections, n_keypoints, dim = keypoints.shape
        keypoints = np.array(keypoints, dtype=float)
        if n_detections != n_bboxes:
            raise ValueError(
                f"Keypoints should have same length as bboxes, got {n_detections} and {n_bboxes}."
            )
        if dim not in [2, 3]:
            raise ValueError(
                f"Keypoints should be of dimension 2 or 3 e.g. [x, y] or [x, y, z], got {dim} dimensions."
            )

    detections = []
    for detection_idx in range(n_bboxes):
        detection = ImgDetectionExtended()
        x_center, y_center, width, height = bboxes[detection_idx]
        detection.x_center = x_center
        detection.y_center = y_center
        detection.width = width
        detection.height = height
        detection.confidence = scores[detection_idx]

        if angles is not None:
            detection.angle = angles[detection_idx]
        if labels is not None:
            detection.label = labels[detection_idx].item()
        if keypoints is not None:
            detection.keypoints = transform_to_keypoints(keypoints[detection_idx])

        detections.append(detection)

    detections_msg = ImgDetectionsExtended()
    detections_msg.detections = detections

    if masks is not None:
        detections_msg.masks = masks

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
