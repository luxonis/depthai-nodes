from typing import List

import numpy as np

from ...messages.creators.keypoints import Keypoint


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
    dim = keypoints.shape[1]

    keypoints_list = []
    for i, keypoint in enumerate(keypoints):
        kp = Keypoint()
        kp.x = keypoint[0]
        kp.y = keypoint[1]

        if dim == 3:
            kp.z = keypoint[2]
        if confidences:
            kp.confidence = confidences[i]

        keypoints_list.append(kp)

    return keypoints_list
