import cv2
import numpy as np

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Converts the input image `arr` (NumPy array) to the planar format expected by depthai.
    The image is resized to the dimensions specified in `shape`.
    
    Parameters:
    - arr: Input NumPy array (image).
    - shape: Target dimensions (width, height).
    
    Returns:
    - A 1D NumPy array with the planar image data.
    """
    if arr.shape[:2] == shape:
        resized = arr 
    else:
        resized = cv2.resize(arr, shape)

    return resized.transpose(2, 0, 1).flatten()
