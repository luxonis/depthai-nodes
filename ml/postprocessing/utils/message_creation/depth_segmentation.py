import depthai as dai
import numpy as np

def create_depth_segmentation_msg(x: np.array, img_frame_type: str) -> dai.ImgFrame:
    """
    Create a message for the segmentation mask or depth image. Input is of the shape (H, W, 1). 
    In the third dimesion we specify the class for segmentation task or depth for depth task.

    Args:
        x (np.array): Input from the segmentation or depth node.
        img_frame_type (str): Type of the image frame. Only 'raw8' and 'raw16' are supported. RAW16 is used for depth task and RAW8 for segmentation task.

    Returns:
        dai.ImgFrame: Output with segmentation classes or depth values.
    """
    if img_frame_type.lower() not in ["raw8", "raw16"]:
        raise ValueError(f"Invalid image frame type: {img_frame_type}. Only 'raw16' and 'raw8' are supported.")
    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(x)
    imgFrame.setWidth(x.shape[1])
    imgFrame.setHeight(x.shape[0])
    if img_frame_type.lower() == "raw8":
        imgFrame.setType(dai.ImgFrame.Type.RAW8)
    elif img_frame_type.lower() == "raw16":
        imgFrame.setType(dai.ImgFrame.Type.RAW16)

    return imgFrame