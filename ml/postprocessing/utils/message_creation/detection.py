import depthai as dai
import numpy as np
from typing import List

def create_detection_message(bboxes: np.ndarray, scores: np.ndarray, labels: List[int] = None) -> dai.ImgDetections:
    """
    Create a message for the detection. The message contains the bounding boxes, labels, and confidence scores of detected objects.
    If there are no labels or we only have one class, we can set labels to None and all detections will have label set to 0.

    Args:
        bboxes (np.ndarray): Detected bounding boxes of shape (N,4) meaning [...,[x_min, y_min, x_max, y_max],...].
        scores (np.ndarray): Confidence scores of detected objects of shape (N,).
        labels (List[int], optional): Labels of detected objects of shape (N,). Defaults to None.

    Returns:
        dai.ImgDetections: Message containing the bounding boxes, labels, and confidence scores of detected objects.
    """

    if not isinstance(bboxes, np.ndarray):
        raise ValueError(f"bboxes should be numpy array, got {type(bboxes)}.")
    if len(bboxes.shape) != 2:
        raise ValueError(f"bboxes should be of shape (N,4) meaning [...,[x_min, y_min, x_max, y_max],...], got {bboxes.shape}.")
    if bboxes.shape[1] != 4:
        raise ValueError(f"bboxes 2nd dimension should be of size 4 e.g. [x_min, y_min, x_max, y_max] got {bboxes.shape[1]}.")
    if not isinstance(scores, np.ndarray):
        raise ValueError(f"scores should be numpy array, got {type(scores)}.")
    if len(scores.shape) != 1:
        raise ValueError(f"scores should be of shape (N,) meaning, got {scores.shape}.")
    if scores.shape[0] != bboxes.shape[0]:
        raise ValueError(f"scores should have same length as bboxes, got {scores.shape[0]} and {bboxes.shape[0]}.")
    if labels is not None:
        if not isinstance(labels, List):
            raise ValueError(f"labels should be list, got {type(labels)}.")
        for label in labels:
            if not isinstance(label, int):
                raise ValueError(f"labels should be list of integers, got {type(label)}.")
        if len(labels) != bboxes.shape[0]:
            raise ValueError(f"labels should have same length as bboxes, got {len(labels)} and {bboxes.shape[0]}.")
    
    if labels is None:
        labels = [0 for _ in range(bboxes.shape[0])]
    
    detections = []
    for bbox, score, label in zip(bboxes, scores, labels):
        detection = dai.ImgDetection()
        detection.confidence = score
        detection.label = label
        detection.xmin = bbox[0]
        detection.ymin = bbox[1]
        detection.xmax = bbox[2]
        detection.ymax = bbox[3]
        detections.append(detection)

    detections_msg = dai.ImgDetections()
    detections_msg.detections = detections
    return detections_msg