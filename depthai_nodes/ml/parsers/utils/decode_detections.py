from typing import Any, Dict, List

import numpy as np


def decode_detections(
    input_size: float,
    stride: int,
    score_threshold: float,
    cls: np.ndarray,
    obj: np.ndarray,
    bbox: np.ndarray,
    kps: np.ndarray,
) -> List[Dict[str, Any]]:
    """Decode the detections from neural network output tensors.

    @param input_size: The input size of the model that produced the detections, (width, height).
    @type input_size: float
    @param stride: The stride used in the detection grid.
    @type stride: int
    @param rows: Number of rows in the detection grid.
    @type rows: int
    @param cols: Number of columns in the detection grid.
    @type cols: int
    @param score_threshold: Minimum score threshold for a detection to be considered valid.
    @type score_threshold: float
    @param cls: 2D array of class scores for each grid cell, shape (grid_size, num_classes).
    @type cls: np.ndarray
    @param obj: 1D array of objectness scores for each grid cell, shape (grid_size,).
    @type obj: np.ndarray
    @param bbox: 2D array of bounding box coordinates, shape (grid_size, 4).
    @type bbox: np.ndarray
    @param kps: 2D array of keypoint coordinates, shape (grid_size, num_keypoints * 2).
    @type kps: np.ndarray

    @return: A list of detections, where each detection is a dictionary containing:
        - "bbox": [x1, y1, width, height] (relative bounding box coordinates)
        - "label": int (class label)
        - "keypoints": List[float] (relative keypoint coordinates)
        - "score": float (detection score)
    @rtype: List[Dict[str, Any]]
    """

    input_width, input_height = input_size
    cols = int(input_size[0] / stride)  # w/stride
    rows = int(input_size[1] / stride)  # h/stride

    # Compute the indices
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    idx = r * cols + c

    # Decode scores
    cls_scores = np.clip(cls[idx], 0, 1)
    obj_scores = np.clip(obj[idx], 0, 1)
    max_cls_scores = np.max(cls_scores, axis=-1)
    scores = np.sqrt(max_cls_scores * obj_scores)

    # Get the labels with the highest score
    labels = np.argmax(cls_scores, axis=-1)

    # Decode bounding boxes
    cx = (c + bbox[idx, 0]) * stride
    cy = (r + bbox[idx, 1]) * stride
    w = np.exp(bbox[idx, 2]) * stride
    h = np.exp(bbox[idx, 3]) * stride
    x1 = cx - w / 2
    y1 = cy - h / 2

    # Decode keypoints
    lx = (kps[idx, ::2] + c[:, :, None]) * stride
    ly = (kps[idx, 1::2] + r[:, :, None]) * stride
    keypoints = np.stack((lx / input_width, ly / input_height), axis=-1)

    # Filter detections based on score_threshold
    mask = scores > score_threshold

    # Boolean indexing
    mask_indices = np.where(mask)
    x1_filtered = x1[mask_indices] / input_width
    y1_filtered = y1[mask_indices] / input_height
    w_filtered = w[mask_indices] / input_width
    h_filtered = h[mask_indices] / input_height
    labels_filtered = labels[mask_indices]
    keypoints_filtered = keypoints[mask_indices]
    scores_filtered = scores[mask_indices]

    # Construct the list of dictionaries
    detections = [
        {
            "bbox": [x1, y1, w, h],
            "label": int(label),
            "keypoints": [
                (x, y) for x, y in keypoints
            ],  # convert keypoints to list of tuples
            "score": float(score),
        }
        for x1, y1, w, h, label, keypoints, score in zip(
            x1_filtered,
            y1_filtered,
            w_filtered,
            h_filtered,
            labels_filtered,
            keypoints_filtered,
            scores_filtered,
        )
    ]

    return detections
