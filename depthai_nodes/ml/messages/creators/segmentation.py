import numpy as np

from ...messages import SegmentationMask


def create_segmentation_message(mask: np.ndarray) -> SegmentationMask:
    """Create a DepthAI message for segmentation mask.

    @param mask: Segmentation map array of shape (H, W) where each value represents a
        segmented object class.
    @type mask: np.array
    @return: Segmentation mask message.
    @rtype: SegmentationMask
    @raise ValueError: If mask is not a numpy array.
    @raise ValueError: If mask is not 2D.
    @raise ValueError: If mask is not of type uint16.
    """

    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(mask)}.")

    if len(mask.shape) != 2:
        raise ValueError(f"Expected 2D input, got {len(mask.shape)}D input.")

    if mask.dtype != np.uint16:
        raise ValueError(f"Expected uint16 input, got {mask.dtype}.")

    mask_msg = SegmentationMask()
    mask_msg.mask = mask

    return mask_msg
