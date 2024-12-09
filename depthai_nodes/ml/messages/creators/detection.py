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
    keypoints_scores: np.ndarray = None,
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
    @param keypoints: Keypoints of detected objects of shape (N, n_keypoints, dim) where
        dim is 2 or 3.
    @type keypoints: Optional[np.array]
    @param keypoints_scores: Confidence scores of detected keypoints of shape (N,
        n_keypoints, 1).
    @type keypoints_scores: Optional[np.ndarray]
    @param masks: Masks of detected objects of shape (H, W).
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
    @raise ValueError: If the keypoints scores are not a numpy array.
    @raise ValueError: If the keypoints scores are not of shape [n_detections,
        n_keypoints, 1].
    @raise ValueError: If the keypoints scores do not have the same length as keypoints.
    @raise ValueError: If the keypoints scores are not between 0 and 1.
    """

    if not isinstance(bboxes, np.ndarray):
        raise ValueError(f"Bounding boxes should be a numpy array, got {type(bboxes)}.")

    if len(bboxes) == 0:
        img_detections = ImgDetectionsExtended()
        if masks is not None:
            img_detections.masks = masks
        return img_detections

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
                f"Labels should have same length as bboxes, got {len(labels)} labels and {n_bboxes} bounding boxes."
            )

    if angles is not None:
        if not isinstance(angles, np.ndarray):
            raise ValueError(f"Angles should be a numpy array, got {type(angles)}.")
        if len(angles) != n_bboxes:
            raise ValueError(
                f"Angles should have same length as bboxes, got {len(angles)} angles and {n_bboxes} bounding boxes."
            )
        for angle in angles:
            if not -360 <= angle <= 360:
                raise ValueError(f"Angles should be between -360 and 360, got {angle}.")

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
                f"Keypoints should have same length as bboxes, got {n_detections} keypoints and {n_bboxes} bounding boxes."
            )
        if dim not in [2, 3]:
            raise ValueError(
                f"Keypoints should be of dimension 2 or 3 e.g. [x, y] or [x, y, z], got {dim} dimensions."
            )

    if keypoints_scores is not None:
        if keypoints is None:
            raise ValueError(
                "Keypoints scores should be provided only if keypoints are provided."
            )
        if not isinstance(keypoints_scores, np.ndarray):
            raise ValueError(
                f"Keypoints scores should be a numpy array, got {type(keypoints_scores)}."
            )

        n_detections, n_keypoints, _ = keypoints.shape
        if keypoints_scores.shape[0] != n_detections:
            raise ValueError(
                f"Keypoints scores should have same length as keypoints, got {len(keypoints_scores)} keypoints scores and {n_detections} keypoints."
            )

        if keypoints_scores.shape[1] != n_keypoints:
            raise ValueError(
                f"Number of keypoints scores per detection should be the same as number of keypoints per detection, got {keypoints_scores.shape[1]} keypoints scores and {n_keypoints} keypoints."
            )

        if not all(0 <= score <= 1 for score in keypoints_scores.flatten()):
            raise ValueError("Keypoints scores should be between 0 and 1.")

    if masks is not None:
        if not isinstance(masks, np.ndarray):
            raise ValueError(f"Masks should be a numpy array, got {type(masks)}.")

        if len(masks.shape) != 2:
            raise ValueError(f"Masks should be of shape (H, W), got {masks.shape}.")

        if masks.dtype != np.int16:
            masks = masks.astype(np.int16)

    detections = []
    for detection_idx in range(n_bboxes):
        detection = ImgDetectionExtended()

        x_center, y_center, width, height = bboxes[detection_idx]
        angle = 0
        if angles is not None:
            angle = float(angles[detection_idx])
        detection.rotated_rect = (
            float(x_center),
            float(y_center),
            float(width),
            float(height),
            angle,
        )
        detection.confidence = float(scores[detection_idx])

        if labels is not None:
            detection.label = int(labels[detection_idx])
        if keypoints is not None:
            if keypoints_scores is not None:
                detection.keypoints = transform_to_keypoints(
                    keypoints[detection_idx], keypoints_scores[detection_idx]
                )
            else:
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
