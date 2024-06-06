import depthai as dai
import numpy as np

def create_depth_message(x: np.array) -> dai.ImgFrame:
    """
    Create a message for the depth image. Input is of the shape (H, W, 1). 
    In the third dimesion we specify the depth in the image.

    Args:
        x (np.array): Input from the depth node.

    Returns:
        dai.ImgFrame: Output depth message in ImgFrame.Type.RAW16.
    """

    if not isinstance(x, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(x)}.")
    if len(x.shape) != 3:
        raise ValueError(f"Expected 3D input, got {len(x.shape)}D input.")
    if x.shape[2] != 1:
        raise ValueError(f"Expected 1 channel in the third dimension, got {x.shape[2]} channels.")
    
    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(x)
    imgFrame.setWidth(x.shape[1])
    imgFrame.setHeight(x.shape[0])
    imgFrame.setType(dai.ImgFrame.Type.RAW16)
    return imgFrame