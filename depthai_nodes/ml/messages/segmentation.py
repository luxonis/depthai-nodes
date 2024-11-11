import depthai as dai
import numpy as np
from numpy.typing import NDArray


class SegmentationMask(dai.Buffer):
    """SegmentationMask class for a single- or multi-object segmentation mask.
    Foreground classes are represented by non-negative integers. Background should be
    represented by the maximum 16-bit integer value (65535).

    Attributes
    ----------
    mask: NDArray[np.uint16]
        Segmentation mask.
    """

    def __init__(self):
        """Initializes the SegmentationMask object."""
        super().__init__()
        self._mask: NDArray[np.uint16] = np.array([])

    @property
    def mask(self) -> NDArray[np.uint16]:
        """Returns the segmentation mask.

        @return: Segmentation mask.
        @rtype: NDArray[np.uint16]
        """
        return self._mask

    @mask.setter
    def mask(self, value: NDArray[np.uint16]):
        """Sets the segmentation mask.

        @param value: Segmentation mask.
        @type value: NDArray[np.uint16])
        @raise TypeError: If value is not a numpy array.
        @raise ValueError: If value is not a 2D numpy array.
        @raise ValueError: If each element is not of type uint16.
        @raise ValueError: If any element is negative.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Mask must be a numpy array.")
        if value.ndim != 2:
            raise ValueError("Mask must be 2D.")
        if value.dtype != np.uint16:
            raise ValueError("Mask must be an array of uint16.")
        if np.any((value < 0)):
            raise ValueError("Mask must be an array of non-negative integers.")
        self._mask = value
