from typing import List, Optional, Tuple

import numpy as np

from depthai_nodes import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
)

from .keypoints import create_keypoints_message


def create_detection_message(
    bboxes: np.ndarray,
    scores: np.ndarray,
    angles: np.ndarray = None,
    labels: np.ndarray = None,
    label_names: Optional[List[str]] = None,
    keypoints: np.ndarray = None,
    keypoints_scores: np.ndarray = None,
    keypoint_label_names: Optional[List[str]] = None,
    keypoint_edges: Optional[List[Tuple[int, int]]] = None,
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
    @param angles: Angles of detected objects expressed in degrees. Defaults to None.
    @type angles: Optional[np.ndarray]
    @param labels: Labels of detected objects of shape (N,). Defaults to None.
    @type labels: Optional[np.ndarray]
    @param label_names: Names of the labels (classes)
    @type label_names: Optional[List[str]]
    @param keypoints: Keypoints of detected objects of shape (N, n_keypoints, dim) where
        dim is 2 or 3. Defaults to None.
    @type keypoints: Optional[np.array]
    @param keypoints_scores: Confidence scores of detected keypoints of shape (N,
        n_keypoints). Defaults to None.
    @type keypoints_scores: Optional[np.ndarray]
    @param keypoint_label_names: Labels of keypoints. Defaults to None.
    @type keypoint_label_names: Optional[List[str]]
    @param keypoint_edges: Connection pairs of keypoints. Defaults to None. Example:
        [(0,1), (1,2), (2,3), (3,0)] shows that keypoint 0 is connected to keypoint 1,
        keypoint 1 is connected to keypoint 2, etc.
    @type keypoint_edges: Optional[List[Tuple[int, int]]]
    @param masks: Masks of detected objects of shape (H, W). Defaults to None.
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

    if label_names is not None:
        if not isinstance(label_names, list):
            raise ValueError(f"Label names should be a list, got {type(label_names)}.")
        if not all(isinstance(label_name, str) for label_name in label_names):
            raise ValueError(f"Label names should be strings, got {label_names}.")

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

    if keypoint_label_names is not None:
        if not isinstance(keypoint_label_names, list):
            raise ValueError(
                f"Keypoint label names should be a list, got {type(keypoint_label_names)}."
            )
        if not all(isinstance(label, str) for label in keypoint_label_names):
            raise ValueError(
                f"Keypoint label names should be a list of strings, got {keypoint_label_names}."
            )

    if keypoint_edges is not None:
        if not isinstance(keypoint_edges, list):
            raise ValueError(
                f"Keypoint edges should be a list, got {type(keypoint_edges)}."
            )
        if not all(
            isinstance(edge, tuple)
            and len(edge) == 2
            and all(isinstance(i, int) for i in edge)
            for edge in keypoint_edges
        ):
            raise ValueError(
                f"Keypoint edges should be a list of tuples of integers, got {keypoint_edges}."
            )

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
            if label_names is not None:
                detection.label_name = label_names[detection_idx]
        if keypoints is not None:
            keypoints_msg = create_keypoints_message(
                keypoints=keypoints[detection_idx],
                scores=(
                    None
                    if keypoints_scores is None
                    else keypoints_scores[detection_idx]
                ),
                label_names=keypoint_label_names,
                edges=keypoint_edges,
            )
            detection.keypoints = keypoints_msg
        detections.append(detection)

    detections_msg = ImgDetectionsExtended()
    detections_msg.detections = detections

    if masks is not None:
        detections_msg.masks = masks

    return detections_msg
