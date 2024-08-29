import depthai as dai
import numpy as np


def create_segmentation_message(x: np.array) -> dai.ImgFrame:
    """Create a DepthAI message for segmentation mask.

    @param x: Segmentation map array of the shape (H, W, E{1}) where E{1} stands for the
        class of the segmented objects.
    @type x: np.array
    @return: Output segmentaion message in ImgFrame.Type.RAW8.
    @rtype: dai.ImgFrame
    @raise ValueError: If the input is not a numpy array.
    @raise ValueError: If the input is not 3D.
    @raise ValueError: If the input 3rd dimension is not E{1}.
    """

    if not isinstance(x, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(x)}.")
    if len(x.shape) != 3:
        raise ValueError(f"Expected 3D input, got {len(x.shape)}D input.")
    if x.shape[2] != 1:
        raise ValueError(
            f"Expected 1 channel in the third dimension, got {x.shape[2]} channels."
        )

    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(x)
    imgFrame.setWidth(x.shape[1])
    imgFrame.setHeight(x.shape[0])
    imgFrame.setType(dai.ImgFrame.Type.RAW8)
    return imgFrame
