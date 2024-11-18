import cv2
import depthai as dai
import numpy as np
from numpy.typing import NDArray


class SegmentationMask(dai.Buffer):
    """SegmentationMask class for a single- or multi-object segmentation mask.
    Unassigned pixels are represented with "-1" and foreground classes with non-negative
    integers.

    Attributes
    ----------
    mask: NDArray[np.int16]
        Segmentation mask.
    """

    def __init__(self):
        """Initializes the SegmentationMask object."""
        super().__init__()
        self._mask: NDArray[np.int16] = np.array([])

    @property
    def mask(self) -> NDArray[np.int16]:
        """Returns the segmentation mask.

        @return: Segmentation mask.
        @rtype: NDArray[np.int16]
        """
        return self._mask

    @mask.setter
    def mask(self, value: NDArray[np.int16]):
        """Sets the segmentation mask.

        @param value: Segmentation mask.
        @type value: NDArray[np.int16])
        @raise TypeError: If value is not a numpy array.
        @raise ValueError: If value is not a 2D numpy array.
        @raise ValueError: If each element is not of type int16.
        @raise ValueError: If any element is smaller than -1.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Mask must be a numpy array.")
        if value.ndim != 2:
            raise ValueError("Mask must be 2D.")
        if value.dtype != np.int16:
            raise ValueError("Mask must be an array of int16.")
        if np.any((value < -1)):
            raise ValueError("Mask must be an array of integers larger or equal to -1.")
        self._mask = value

    def getVisualizationMessage(self) -> dai.ImgFrame:
        img_frame = dai.ImgFrame()
        mask = self._mask.copy()

        unique_values = np.unique(mask[mask >= 0])
        scaled_mask = np.zeros_like(mask, dtype=np.uint8)

        if unique_values.size == 0:
            print("no classes found")
            return img_frame.setCvFrame(scaled_mask, dai.ImgFrame.Type.BGR888i)

        min_val, max_val = unique_values.min(), unique_values.max()

        scaled_mask = ((mask - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        scaled_mask[mask == -1] = 0
        colored_mask = cv2.applyColorMap(scaled_mask, cv2.COLORMAP_RAINBOW)
        colored_mask[mask == -1] = [0, 0, 0]

        return img_frame.setCvFrame(colored_mask, dai.ImgFrame.Type.BGR888i)
