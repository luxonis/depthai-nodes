import depthai as dai
import numpy as np
from numpy.typing import NDArray


class SegmentationMask(dai.Buffer):
    """SegmentationMask class for a single- or multi-object segmentation mask.
    Background is represented with "0" and foreground classes with positive integers.

    Attributes
    ----------
    mask: NDArray[np.uint8]
        Segmentation mask.
    """

    def __init__(self):
        """Initializes the SegmentationMask object."""
        super().__init__()
        self._mask: NDArray[np.uint8] = np.array([])

    @property
    def mask(self) -> NDArray[np.uint8]:
        """Returns the segmentation mask.

        @return: Segmentation mask.
        @rtype: NDArray[np.uint8]
        """
        return self._mask

    @mask.setter
    def mask(self, value: NDArray[np.int8]):
        """Sets the segmentation mask.

        @param value: Segmentation mask.
        @type value: NDArray[np.int8])
        @raise TypeError: If value is not a numpy array.
        @raise ValueError: If value is not a 2D numpy array.
        @raise ValueError: If each element is not of type int8.
        @raise ValueError: If each element is larger or equal to -1.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Mask must be a numpy array.")
        if value.ndim != 2:
            raise ValueError("Mask must be 2D.")
        if value.dtype != np.uint8:
            raise ValueError("Mask must be an array of uint8.")
        if np.any((value < -1)):
            raise ValueError("Mask must be an array of non-negative integers.")
        self._mask = value
