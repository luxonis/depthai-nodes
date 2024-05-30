import depthai as dai
import numpy as np

def create_segmentation_message(x: np.array) -> dai.ImgFrame:
    """
    Create a message for the segmentation node output. Input is of the shape (H, W, 1). 
    In the third dimesion we specify the class of the segmented objects.

    Args:
        x (np.array): Input from the segmentation node.

    Returns:
        dai.ImgFrame: Output segmentaion message in ImgFrame.Type.RAW8.
    """
    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(x)
    imgFrame.setWidth(x.shape[1])
    imgFrame.setHeight(x.shape[0])
    imgFrame.setType(dai.ImgFrame.Type.RAW8)
    return imgFrame