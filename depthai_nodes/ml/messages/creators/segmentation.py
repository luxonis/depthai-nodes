from typing import List, Union

import depthai as dai
import numpy as np

from ...messages import SegmentationMasks


def create_segmentation_message(x: np.ndarray) -> dai.ImgFrame:
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
    if isinstance(x[0, 0, 0], float):
        raise ValueError(f"Expected int type, got {type(x[0, 0, 0])}.")
    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(x)
    imgFrame.setWidth(x.shape[1])
    imgFrame.setHeight(x.shape[0])
    imgFrame.setType(dai.ImgFrame.Type.RAW8)
    return imgFrame


def create_sam_message(x: Union[np.ndarray, List[np.ndarray]]) -> SegmentationMasks:
    """Create a DepthAI message for segmentation masks.

    @param x: List of segmentation map arrays of the shape (N, H, W).
    @type x: Union[np.array, List[np.array]]
    @return: Output segmentaion message in SegmentationMasks.
    @rtype: SegmentationMasks
    @raise ValueError: If the input is not a numpy array or list of numpy arrays.
    @raise ValueError: If the input is not 3D.
    """
    if not isinstance(x, (np.ndarray, list)):
        raise ValueError(f"Expected numpy array or list, got {type(x)}.")
    for mask in x:
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(mask)}.")
        if len(mask.shape) != 2:
            raise ValueError(f"Expected 2D input, got {len(mask.shape)}D input.")

    masks_msg = SegmentationMasks()
    if len(x) != 0:
        masks_msg.masks = x if isinstance(x, np.ndarray) else np.array(x)
    return masks_msg
