import numpy as np
from depthai import SegmentationMask


def create_segmentation_message(mask: np.ndarray) -> SegmentationMask:
    """Create a DepthAI message for segmentation mask.

    @param mask: Segmentation map array of shape (H, W) where each value represents a
        segmented object class. Index 255 represents background.
    @type mask: np.array
    @return: Segmentation mask message.
    @rtype: SegmentationMask
    @raise ValueError: If mask is not a numpy array.
    @raise ValueError: If mask is not 2D.
    @raise ValueError: If mask is not of type uint8.
    """

    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(mask)}.")

    if len(mask.shape) != 2:
        raise ValueError(f"Expected 2D input, got {len(mask.shape)}D input.")

    if mask.dtype != np.uint8:
        raise ValueError(f"Expected uint8 input, got {mask.dtype}.")

    mask_msg = SegmentationMask()
    mask_msg.setCvMask(mask)

    return mask_msg
