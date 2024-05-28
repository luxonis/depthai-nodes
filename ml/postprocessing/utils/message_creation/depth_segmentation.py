import depthai as dai
import numpy as np

def create_segmentation_msg(mask: np.array) -> dai.ImgFrame:
    """
    Create a message for the segmentation mask. Mask is of the shape (H, W, 1). In the third dimesion we specify the class.

    Args:
        mask (np.array): The segmentation mask.

    Returns:
        dai.ImgFrame: The message containing the segmentation mask.
    """
    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(mask)
    imgFrame.setWidth(mask.shape[1])
    imgFrame.setHeight(mask.shape[0])
    imgFrame.setType(dai.ImgFrame.Type.GRAY8)

    return imgFrame