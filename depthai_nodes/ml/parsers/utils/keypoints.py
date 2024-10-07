import numpy as np


def normalize_keypoints(keypoints: np.ndarray, height: int, width: int):
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

    if not isinstance(keypoints, np.ndarray):
        raise TypeError("Keypoints must be a numpy array.")

    if keypoints.shape[1] != 2:
        raise ValueError(
            "Keypoints must be of shape (N, 2). Other options are currently not supported"
        )
    keypoints[:, 0] = keypoints[:, 0] / width
    keypoints[:, 1] = keypoints[:, 1] / height

    return keypoints
