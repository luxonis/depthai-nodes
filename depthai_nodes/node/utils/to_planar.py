from typing import Tuple

import cv2
import numpy as np


def to_planar(arr: np.ndarray, shape: Tuple) -> np.ndarray:
    """Converts the input image `arr` (NumPy array) to the planar format expected by
    depthai. The image is resized to the dimensions specified in `shape`.

    @param arr: Input NumPy array (image).
    @type arr: np.ndarray
    @param shape: Target dimensions (width, height).
    @type shape: tuple
    @return: A 1D NumPy array with the planar image data.
    @rtype: np.ndarray
    """
    if np.array_equal(arr.shape[:2], shape):
        resized = arr
    else:
        resized = cv2.resize(arr, shape)

    return resized.transpose(2, 0, 1).flatten()
