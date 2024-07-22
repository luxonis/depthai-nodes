from typing import List, Union

import depthai as dai
import numpy as np

from ...messages import HandKeypoints, Keypoints


def create_hand_keypoints_message(
    hand_keypoints: np.ndarray,
    handedness: float,
    confidence: float,
    confidence_threshold: float,
) -> HandKeypoints:
    """Create a DepthAI message for hand keypoints detection.

    @param hand_keypoints: Detected 3D hand keypoints of shape (N,3) meaning [...,[x, y, z],...].
    @type hand_keypoints: np.ndarray
    @param handedness: Handedness score of the detected hand (left: < 0.5, right > 0.5).
    @type handedness: float
    @param confidence: Confidence score of the detected hand.
    @type confidence: float
    @param confidence_threshold: Confidence threshold for the present hand.
    @type confidence_threshold: float

    @return: HandKeypoints message containing the detected hand keypoints, handedness, and confidence score.
    @rtype: HandKeypoints

    @raise ValueError: If the hand_keypoints are not a numpy array.
    @raise ValueError: If the hand_keypoints are not of shape (N,3).
    @raise ValueError: If the hand_keypoints 2nd dimension is not of size E{3}.
    @raise ValueError: If the handedness is not a float.
    @raise ValueError: If the confidence is not a float.
    """

    if not isinstance(hand_keypoints, np.ndarray):
        raise ValueError(
            f"hand_keypoints should be numpy array, got {type(hand_keypoints)}."
        )
    if len(hand_keypoints.shape) != 2:
        raise ValueError(
            f"hand_keypoints should be of shape (N,3) meaning [...,[x, y, z],...], got {hand_keypoints.shape}."
        )
    if hand_keypoints.shape[1] != 3:
        raise ValueError(
            f"hand_keypoints 2nd dimension should be of size 3 e.g. [x, y, z], got {hand_keypoints.shape[1]}."
        )
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


def create_keypoints_message(
    keypoints: Union[np.ndarray, List[List[float]]],
    scores: Union[np.ndarray, List[float]] = None,
    confidence_threshold: float = None,
) -> Keypoints:
    """Create a DepthAI message for the keypoints.

    @param keypoints: Detected 2D or 3D keypoints of shape (N,2 or 3) meaning [...,[x, y],...] or [...,[x, y, z],...].
    @type keypoints: np.ndarray or List[List[float]]
    @param scores: Confidence scores of the detected keypoints.
    @type scores: np.ndarray or List[float]
    @param confidence_threshold: Confidence threshold of keypoint detections.
    @type confidence_threshold: float

    @return: Keypoints message containing the detected keypoints.
    @rtype: Keypoints

    @raise ValueError: If the keypoints are not a numpy array or list.
    @raise ValueError: If the keypoints are not of shape (N,2 or 3).
    @raise ValueError: If the keypoints 2nd dimension is not of size E{2} or E{3}.
    @raise ValueError: If the scores are not a numpy array or list.
    @raise ValueError: If the scores are not of shape (N,).
    @raise ValueError: If the keypoints and scores do not have the same length.
    @raise ValueError: If the confidence threshold is not a float.
    @raise ValueError: If the confidence threshold is not provided when scores are provided.
    """

    if not isinstance(keypoints, np.ndarray):
        if not isinstance(keypoints, list):
            raise ValueError(
                f"keypoints should be numpy array or list, got {type(keypoints)}."
            )
        for keypoint in keypoints:
            if not isinstance(keypoint, list):
                raise ValueError(
                    f"keypoints should be list of lists or np.array, got list of {type(keypoint)}."
                )
            if len(keypoint) not in [2, 3]:
                raise ValueError(
                    f"keypoints inner list should be of size 2 or 3 e.g. [x, y] or [x, y, z], got {len(keypoint)}."
                )
            for coord in keypoint:
                if not isinstance(coord, (float)):
                    raise ValueError(
                        f"keypoints inner list should contain only float, got {type(coord)}."
                    )
        keypoints = np.array(keypoints)
    if len(keypoints.shape) != 2:
        raise ValueError(
            f"keypoints should be of shape (N,2 or 3) got {keypoints.shape}."
        )
    if keypoints.shape[1] not in [2, 3]:
        raise ValueError(
            f"keypoints 2nd dimension should be of size 2 or 3 e.g. [x, y] or [x, y, z], got {keypoints.shape[1]}."
        )
    if scores is not None:
        if not isinstance(scores, np.ndarray):
            if not isinstance(scores, list):
                raise ValueError(
                    f"scores should be numpy array or list, got {type(scores)}."
                )
            for score in scores:
                if not isinstance(score, float):
                    raise ValueError(
                        f"scores should be list of floats or np.array, got list of {type(score)}."
                    )
            scores = np.array(scores)
        if len(scores.shape) != 1:
            raise ValueError(
                f"scores should be of shape (N,) meaning [...,score,...], got {scores.shape}."
            )
        if keypoints.shape[0] != scores.shape[0]:
            raise ValueError(
                f"keypoints and scores should have the same length, got {keypoints.shape[0]} and {scores.shape[0]}."
            )
        if confidence_threshold is None:
            raise ValueError(
                "confidence_threshold should be provided when scores are provided."
            )
    if confidence_threshold is not None:
        if not isinstance(confidence_threshold, float):
            raise ValueError(
                f"confidence_threshold should be float, got {type(confidence_threshold)}."
            )
        if scores is None:
            raise ValueError(
                "confidence_threshold should be provided when scores are provided."
            )

    use_3d = keypoints.shape[1] == 3

    keypoints_msg = Keypoints()
    points = []
    for i, keypoint in enumerate(keypoints):
        if scores is not None:
            if scores[i] < confidence_threshold:
                continue
        pt = dai.Point3f()
        pt.x = keypoint[0]
        pt.y = keypoint[1]
        pt.z = keypoint[2] if use_3d else 0
        points.append(pt)

    keypoints_msg.keypoints = points
    return keypoints_msg
