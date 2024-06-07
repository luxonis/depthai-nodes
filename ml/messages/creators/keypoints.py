import depthai as dai
import numpy as np
from typing import List
from ...messages import HandKeypoints, Keypoints

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

def create_animal_keypoints_message(animal_keypoints: np.ndarray, scores: np.ndarray, confidence_threshold: float) -> Keypoints:
    """
    Create a message for the animal keypoint detection. The message contains the coordinates of maximum 39 detected animal keypoints.

    Args:
        animal_keypoints (np.ndarray): Detected animal keypoints of shape (N,2) meaning [...,[x, y],...].
        scores (np.ndarray): Confidence scores of the detected animal keypoints.
        confidence_threshold (float): Confidence threshold for the keypoints.

    Returns:
        Keypoints: Message containing the 3D coordinates of the detected animal keypoints.
    """

    if not isinstance(animal_keypoints, np.ndarray):
        raise ValueError(f"animal_keypoints should be numpy array, got {type(animal_keypoints)}.")
    if len(animal_keypoints.shape) != 2:
        raise ValueError(f"animal_keypoints should be of shape (N,2) meaning [...,[x, y],...], got {animal_keypoints.shape}.")
    if animal_keypoints.shape[1] != 2:
        raise ValueError(f"animal_keypoints 2nd dimension should be of size 2 e.g. [x, y], got {animal_keypoints.shape[1]}.")
    if not isinstance(scores, np.ndarray):
        raise ValueError(f"scores should be numpy array, got {type(scores)}.")
    if len(scores.shape) != 1:
        raise ValueError(f"scores should be of shape (N,) meaning [...,score,...], got {scores.shape}.")
    if not isinstance(confidence_threshold, float):
        raise ValueError(f"confidence_threshold should be float, got {type(confidence_threshold)}.")

    animal_keypoints_msg = Keypoints()
    points = []
    for keypoint, score in zip(animal_keypoints, scores):
        if score >= confidence_threshold:
            pt = dai.Point3f()
            pt.x = keypoint[0]
            pt.y = keypoint[1]
            pt.z = 0
            points.append(pt)

    animal_keypoints_msg.keypoints = points

    return animal_keypoints_msg