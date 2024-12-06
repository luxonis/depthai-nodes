from typing import List

import numpy as np

from ...messages.creators.keypoints import Keypoint


def normalize_keypoints(keypoints: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize keypoint coordinates to (0, 1).

    Parameters:
    @param keypoints: A numpy array of shape (N, 2) or (N, K, 2) where N is the number of keypoint sets and K is the number of keypoint in each set.
    @type np.ndarray
    @param height: The height of the image.
    @type height: int
    @param width: The width of the image.
    @type width: int

    Returns:
    np.ndarray: A numpy array of shape (N, 2) containing the normalized keypoints.
    """
    keypoints = keypoints.astype(np.float32)
    if not isinstance(keypoints, np.ndarray):
        raise TypeError("Keypoints must be a numpy array.")

    if len(keypoints.shape) != 2 and len(keypoints.shape) != 3:
        raise ValueError(
            f"Keypoints must be of shape (N, 2) or (N, K, 2). Got {keypoints.shape}."
        )

    if keypoints.shape[1] != 2 and len(keypoints.shape) == 2:
        raise ValueError(
            "Keypoints must be of shape (N, 2). Other options are currently not supported"
        )
    elif len(keypoints.shape) == 3:
        if keypoints.shape[2] != 2:
            raise ValueError(
                "Keypoints must be of shape (N, K, 2). Other options are currently not supported"
            )
    keypoints[:, 0] = keypoints[:, 0] / width
    keypoints[:, 1] = keypoints[:, 1] / height

    return keypoints


def transform_to_keypoints(
    keypoints: np.ndarray, confidences: np.ndarray = None
) -> List[Keypoint]:
    """Transforms an array representing keypoints into a list of Keypoint objects.

    @param keypoints: Detected 2D or 3D keypoints of shape (N,2 or 3) meaning [...,[x, y],...] or [...,[x, y, z],...].
    @type keypoints: np.ndarray
    @param confidences: Confidence scores of the detected keypoints.
    @type confidences: Optional[np.ndarray]

    @return: List of Keypoint objects.
    @rtype: List[Keypoint]
    """
    if not isinstance(keypoints, np.ndarray):
        raise ValueError("Keypoints must be a numpy array.")
    if len(keypoints.shape) != 2:
        raise ValueError(
            f"Keypoints must be of shape (N, 2). Got shape {keypoints.shape}."
        )
    if keypoints.shape[1] != 2 and keypoints.shape[1] != 3:
        raise ValueError(
            f"Keypoints must be of shape (N, 2) or (N, 3). Got shape {keypoints.shape}."
        )

    if confidences is not None:
        if not isinstance(confidences, np.ndarray):
            raise ValueError("Confidences must be a numpy array.")
        if len(confidences.shape) != 1:
            raise ValueError(
                f"Confidences must be of shape (N,). Got shape {confidences.shape}."
            )
        if len(confidences) != len(keypoints):
            raise ValueError(
                f"Confidences should have same length as keypoints, got {len(confidences)} confidences and {len(keypoints)} keypoints."
            )

    dim = keypoints.shape[1]

    keypoints_list = []
    for i, keypoint in enumerate(keypoints):
        kp = Keypoint()
        kp.x = keypoint[0]
        kp.y = keypoint[1]

        if dim == 3:
            kp.z = keypoint[2]
        if confidences is not None:
            kp.confidence = confidences[i]

        keypoints_list.append(kp)

    return keypoints_list
