import depthai as dai
import numpy as np
from numpy.typing import NDArray


class SegmentationMask(dai.Buffer):
    """SegmentationMask class for storing a single segmentation mask. Only background
    (0) and single foreground class (1) are supported.

    Attributes
    ----------
    mask: NDArray[np.int8]
        Segmentation mask.
    """

    def __init__(self):
        """Initializes the SegmentationMask object."""
        super().__init__()
        self._mask: NDArray[np.int8] = np.array([])

    @property
    def mask(self) -> NDArray[np.int8]:
        """Returns the segmentation mask.

        @return: Segmentation mask.
        @rtype: NDArray[np.int8]
        """
        return self._masks

    @mask.setter
    def mask(self, value: NDArray[np.int8]):
        """Sets the segmentation mask.

        @param value: Segmentation mask.
        @type value: NDArray[np.int8])
        @raise TypeError: If value is not a numpy array.
        @raise ValueError: If value is not a 2D numpy array.
        @raise ValueError: If each element is not of type int8.
        @raise ValueError: If any element not in the range [0, 1].
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Mask must be a numpy array.")
        if value.ndim != 2:
            raise ValueError("Mask must be 2D.")
        if value.dtype != np.int8:
            raise ValueError("Mask must be an array of int8.")
        if np.any((value < 0)) or np.any((value > 1)):
            raise ValueError("Mask must be an array of non-negative int8 values.")
        self._masks = value


class SegmentationMasks(dai.Buffer):
    """SegmentationMasks class for storing multiple segmentation masks. Background (0)
    and multiple foreground classes (ints >= 1) are supported.

    Attributes
    ----------
    masks: NDArray[np.int8]
        Segmentation masks.
    """

    def __init__(self):
        """Initializes the SegmentationMask object."""
        super().__init__()
        self._masks: NDArray[np.int8] = np.array([])

    @property
    def masks(self) -> NDArray[np.int8]:
        """Returns the segmentation masks.

        @return: Segmentation masks.
        @rtype: NDArray[np.int8]
        """
        return self._masks

    @masks.setter
    def masks(self, value: NDArray[np.int8]):
        """Sets the segmentation masks.

        @param value: Segmentation masks.
        @type value: NDArray[np.int8])
        @raise TypeError: If value is not a numpy array.
        @raise ValueError: If value is not a 2D numpy array.
        @raise ValueError: If each element is not of type int8.
        @raise ValueError: If any element is negative.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Masks must be a numpy array.")
        if value.ndim != 2:
            raise ValueError("Masks must be 2D.")
        if value.dtype != np.int8:
            raise ValueError("Masks must be an array of int8.")
        if np.any((value < 0)):
            raise ValueError(
                "Masks must be an array of non-negative values."
            )
        self._masks = value


class SegmentationMasksSAM(dai.Buffer):
    """SegmentationMasksSAM class for storing segmentation masks.
    TODO: remove this message and use SegmentationMasks instead

    Attributes
    ----------
    masks: np.ndarray
        Mask coefficients.
    """

    def __init__(self):
        """Initializes the SegmentationMasks object."""
        super().__init__()
        self._masks: np.ndarray = np.array([])

    @property
    def masks(self) -> np.ndarray:
        """Returns the masks coefficients.

        @return: Masks coefficients.
        @rtype: np.ndarray
        """
        return self._masks

    @masks.setter
    def masks(self, value: np.ndarray):
        """Sets the masks coefficients.

        @param value: Masks coefficients.
        @type value: np.ndarray
        @raise TypeError: If the masks is not a numpy array.
        @raise ValueError: If the masks is not 3D.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Masks must be a numpy array.")
        if value.ndim != 3:
            raise ValueError("Masks must be 3D.")
        self._masks = value
