import depthai as dai
import numpy as np
from typing import List
from ...messages import HandKeypoints

def create_hand_keypoints_message(hand_keypoints: np.ndarray, handedness: float, confidence: float, confidence_threshold: float) -> HandKeypoints:
    """
    Create a message for the hand keypoint detection. The message contains the 3D coordinates of the detected hand keypoints, handedness, and confidence score.

    Args:
        hand_keypoints (np.ndarray): Detected hand keypoints of shape (N,3) meaning [...,[x, y, z],...].
        handedness (float): Handedness score of the detected hand (left or right).
        confidence (float): Confidence score of the detected hand.
        confidence_threshold (float): Confidence threshold for the overall hand.

    Returns:
        HandKeypoints: Message containing the 3D coordinates of the detected hand keypoints, handedness, and confidence score.
    """

    if not isinstance(hand_keypoints, np.ndarray):
        raise ValueError(f"hand_keypoints should be numpy array, got {type(hand_keypoints)}.")
    if len(hand_keypoints.shape) != 2:
        raise ValueError(f"hand_keypoints should be of shape (N,3) meaning [...,[x, y, z],...], got {hand_keypoints.shape}.")
    if hand_keypoints.shape[1] != 3:
        raise ValueError(f"hand_keypoints 2nd dimension should be of size 3 e.g. [x, y, z], got {hand_keypoints.shape[1]}.")
    if not isinstance(handedness, float):
        raise ValueError(f"handedness should be float, got {type(handedness)}.")
    if not isinstance(confidence, float):
        raise ValueError(f"confidence should be float, got {type(confidence)}.")
    
    hand_keypoints_msg = HandKeypoints()
    hand_keypoints_msg.handedness = handedness
    hand_keypoints_msg.confidence = confidence
    points = []
    if confidence >= confidence_threshold:
        for i in range(hand_keypoints.shape[0]):
            pt = dai.Point3f()
            pt.x = hand_keypoints[i][0]
            pt.y = hand_keypoints[i][1]
            pt.z = hand_keypoints[i][2]
            points.append(pt)
    hand_keypoints_msg.keypoints = points

    return hand_keypoints_msg