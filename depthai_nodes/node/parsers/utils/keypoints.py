import numpy as np


def normalize_keypoints(keypoints: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize keypoint coordinates to (0, 1).

    Args:
        keypoints: A numpy array of shape (N, 2) or (N, K, 2) where N is the number of
            keypoint sets and K is the number of keypoint in each set.
        height: The height of the image.
        width: The width of the image.
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


def compute_keypoints(
    keypoints: np.ndarray,
    *,
    n_keypoints: int,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Reshape and normalize a keypoint tensor.

    Args:
        keypoints: Model keypoint tensor.
        n_keypoints: Number of keypoints encoded per prediction.
        scale_factor: Nonzero divisor used to convert model coordinates to normalized
            coordinates.

    Returns:
        Float32 coordinates of shape ``(n_keypoints, 2)`` or ``(n_keypoints, 3)``,
        divided by ``scale_factor`` and clipped to [0, 1].

    Raises:
        ValueError: If the tensor does not contain two or three coordinates per
            keypoint.
    """
    parsed_keypoints = np.asarray(keypoints, dtype=np.float32)
    num_coords = int(np.prod(parsed_keypoints.shape) / n_keypoints)

    if num_coords not in [2, 3]:
        raise ValueError(f"Expected 2 or 3 coordinates per keypoint, got {num_coords}.")

    parsed_keypoints = parsed_keypoints.reshape(n_keypoints, num_coords)
    parsed_keypoints /= scale_factor
    return np.clip(parsed_keypoints, 0, 1)
