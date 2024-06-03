import depthai as dai
import numpy as np
from typing import List
from ....messages import HandKeypoints

def create_hand_keypoints_message(hand_keypoints: np.ndarray, handdedness: float, confidence: float, confidence_threshold: float) -> HandKeypoints:
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

    if not isinstance(hand_keypoints, np.ndarray):
        raise ValueError(f"hand_keypoints should be numpy array, got {type(hand_keypoints)}.")
    if len(hand_keypoints.shape) != 2:
        raise ValueError(f"hand_keypoints should be of shape (N,3) meaning [...,[x, y, z],...], got {hand_keypoints.shape}.")
    if hand_keypoints.shape[1] != 3:
        raise ValueError(f"hand_keypoints 2nd dimension should be of size 3 e.g. [x, y, z], got {hand_keypoints.shape[1]}.")
    if not isinstance(handdedness, float):
        raise ValueError(f"handdedness should be float, got {type(handdedness)}.")
    if not isinstance(confidence, float):
        raise ValueError(f"confidence should be float, got {type(confidence)}.")
    
    hand_keypoints_msg = HandKeypoints()
    hand_keypoints_msg.handdedness = handdedness
    hand_keypoints_msg.confidence = confidence
    points = []
    if confidence >= confidence_threshold:
        for i in range(hand_keypoints.shape[0]):
            pt = dai.Point3f()
            pt.x = hand_keypoints[i][0]
            pt.y = hand_keypoints[i][1]
            pt.z = hand_keypoints[i][2]
            points.append(pt)
    hand_keypoints_msg.landmarks = points

    return hand_keypoints_msg