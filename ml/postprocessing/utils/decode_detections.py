import numpy as np
from typing import List, Dict, Any


def decode_detections(
    input_size: float,
    stride: int,
    score_threshold: float,
    cls: np.ndarray,
    obj: np.ndarray,
    bbox: np.ndarray,
    kps: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Decode the detections from neural network output tensors.

    Args:
        input_size (float): The input size of the model that produced the detections, (width, height).
        stride (int): The stride used in the detection grid.
        rows (int): Number of rows in the detection grid.
        cols (int): Number of columns in the detection grid.
        score_threshold (float): Minimum score threshold for a detection to be considered valid.
        cls (np.ndarray): 2D array of class scores for each grid cell, shape (grid_size, num_classes).
        obj (np.ndarray): 1D array of objectness scores for each grid cell, shape (grid_size,).
        bbox (np.ndarray): 2D array of bounding box coordinates, shape (grid_size, 4).
        kps (np.ndarray): 2D array of keypoint coordinates, shape (grid_size, num_keypoints * 2).

    Returns:
        List[Dict[str, Any]]: A list of detections, where each detection is a dictionary containing:
            - "bbox": [x1, y1, width, height] (relative bounding box coordinates)
            - "label": int (class label)
            - "keypoints": List[float] (relative keypoint coordinates)
            - "score": float (detection score)
    """

    input_width, input_height = input_size
    cols = int(input_size[0] / stride)  # w/stride
    rows = int(input_size[1] / stride)  # h/stride
    num_keypoints = kps.shape[1] // 2
    detections = []

    # Iterate over each grid cell
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c

            # Decode scores
            cls_scores = np.clip(cls[idx], 0, 1)
            obj_score = np.clip(obj[idx], 0, 1)
            max_cls_score = np.max(cls_scores)
            score = np.sqrt(max_cls_score * obj_score)

            # Get the label with the highest score
            label = np.argmax(cls_scores)

            # Decode bounding box
            cx = (c + bbox[idx, 0]) * stride
            cy = (r + bbox[idx, 1]) * stride
            w = np.exp(bbox[idx, 2]) * stride
            h = np.exp(bbox[idx, 3]) * stride
            x1 = cx - w / 2
            y1 = cy - h / 2

            # Decode keypoints
            keypoints = []
            for n in range(num_keypoints):
                lx = (kps[idx, 2 * n] + c) * stride
                ly = (kps[idx, 2 * n + 1] + r) * stride
                keypoints.append((lx/input_width, ly/input_height))

            # Append detection result
            if score > score_threshold:
                detections.append(
                    {
                "bbox": [x1/input_width, y1/input_height, w/input_width, h/input_height],
                "label": int(label),
                "keypoints": keypoints,
                "score": float(score),
            })

    return detections
