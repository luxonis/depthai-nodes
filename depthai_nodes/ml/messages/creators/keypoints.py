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
    @raise ValueError: If the confidence_threshold is not a float.
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
        raise ValueError(f"Handedness should be float, got {type(handedness)}.")
    if not isinstance(confidence, float):
        raise ValueError(f"Confidence should be float, got {type(confidence)}.")
    if not isinstance(confidence_threshold, float):
        raise ValueError(
            f"Confidence threshold should be float, got {type(confidence_threshold)}."
        )

    if handedness < 0 or handedness > 1:
        raise ValueError(
            f"Handedness should be between 0 and 1, got handedness {handedness}."
        )

    if confidence < 0 or confidence > 1:
        raise ValueError(
            f"Confidence should be between 0 and 1, got confidence {confidence}."
        )
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError(
            f"Confidence threshold should be between 0 and 1, got confidence threshold {confidence_threshold}."
        )

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
    @type scores: Optional[np.ndarray or List[float]]
    @param confidence_threshold: Confidence threshold of keypoint detections.
    @type confidence_threshold: Optional[float]

    @return: Keypoints message containing the detected keypoints.
    @rtype: Keypoints

    @raise ValueError: If the keypoints are not a numpy array or list.
    @raise ValueError: If the scores are not a numpy array or list.
    @raise ValueError: If scores and keypoints do not have the same length.
    @raise ValueError: If score values are not floats.
    @raise ValueError: If score values are not between 0 and 1.
    @raise ValueError: If the confidence threshold is not a float.
    @raise ValueError: If the confidence threshold is not between 0 and 1.
    @raise ValueError: If the keypoints are not of shape (N,2 or 3).
    @raise ValueError: If the keypoints 2nd dimension is not of size E{2} or E{3}.
    """

    if not isinstance(keypoints, (np.ndarray, list)):
        raise ValueError(
            f"Keypoints should be numpy array or list, got {type(keypoints)}."
        )

    if scores is not None:
        if not isinstance(scores, (np.ndarray, list)):
            raise ValueError(
                f"Scores should be numpy array or list, got {type(scores)}."
            )

        if len(keypoints) != len(scores):
            raise ValueError(
                "Keypoints and scores should have the same length. Got {} keypoints and {} scores.".format(
                    len(keypoints), len(scores)
                )
            )

        if not all(isinstance(score, (float, np.floating)) for score in scores):
            raise ValueError("Scores should only contain float values.")
        if not all(0 <= score <= 1 for score in scores):
            raise ValueError("Scores should only contain values between 0 and 1.")

    if confidence_threshold is not None:
        if not isinstance(confidence_threshold, float):
            raise ValueError(
                f"The confidence_threshold should be float, got {type(confidence_threshold)}."
            )

        if not (0 <= confidence_threshold <= 1):
            raise ValueError(
                f"The confidence_threshold should be between 0 and 1, got confidence_threshold {confidence_threshold}."
            )

    if len(keypoints) != 0:
        dimension = len(keypoints[0])
        if dimension != 2 and dimension != 3:
            raise ValueError(
                f"All keypoints should be of dimension 2 or 3, got dimension {dimension}."
            )

    if isinstance(keypoints, list):
        for keypoint in keypoints:
            if not isinstance(keypoint, list):
                raise ValueError(
                    f"Keypoints should be list of lists or np.array, got list of {type(keypoint)}."
                )
            if len(keypoint) != dimension:
                raise ValueError(
                    "All keypoints have to be of same dimension e.g. [x, y] or [x, y, z], got mixed inner dimensions."
                )
            for coord in keypoint:
                if not isinstance(coord, (float, np.floating)):
                    raise ValueError(
                        f"Keypoints inner list should contain only float, got {type(coord)}."
                    )

    keypoints = np.array(keypoints)
    if scores is not None:
        scores = np.array(scores)

    if len(keypoints) != 0:
        if len(keypoints.shape) != 2:
            raise ValueError(
                f"Keypoints should be of shape (N,2 or 3) got {keypoints.shape}."
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
