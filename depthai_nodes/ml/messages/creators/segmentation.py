import numpy as np

from ...messages import SegmentationMask


def create_segmentation_message(mask: np.ndarray) -> SegmentationMask:
    """Create a DepthAI message for segmentation mask.

    @param mask: Segmentation map array of shape (H, W) where each value represents a
        segmented object class.
    @type mask: np.array
    @return: Segmentaion mask message.
    @rtype: SegmentationMask
    @raise ValueError: If mask is not a numpy array.
    @raise ValueError: If mask is not 2D.
    @raise ValueError: If mask values are not integers.
    """

    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(mask)}.")

    if len(mask.shape) != 2:
        raise ValueError(f"Expected 2D input, got {len(mask.shape)}D input.")

    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(
            f"Unexpected mask type. Expected an array of integers, got {mask.dtype}."
        )

    mask_msg = SegmentationMask()
    mask_msg.mask = mask

    return mask_msg
