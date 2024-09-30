import numpy as np


def normalize_keypoint(keypoint: np.ndarray, height: int, width: int):
    """Normalize keypoint coordinates to (0, 1).

    Parameters:
    @param keypoint: A tuple or list with 2 elements (x, y).
    @type np.ndarray
    @param height: The height of the image.
    @type height: int
    @param width: The width of the image.
    @type width: int

    Returns:
    tuple: A list with 2 elements [x, y].
    """

    if len(keypoint) != 2:
        raise ValueError(
            "Keypoint must be a tuple or list of length 2. Other options are currently not supported."
        )

    x, y = keypoint
    return [x / width, y / height]


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

    # TODO: expect keypoints to be either of shape (N,2) - one keypoint set or (batch,N,2) - multiple keypoint sets

    if not isinstance(keypoints, np.ndarray):
        raise TypeError("Keypoints must be a numpy array.")

    return np.array(
        [
            [normalize_keypoint(keypoint, height, width) for keypoint in keypoint_set]
            for keypoint_set in keypoints.tolist()
        ]
    )
