import depthai as dai
import numpy as np


class SegmentationMasks(dai.Buffer):
    """SegmentationMasks class for storing segmentation masks.

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
    def masks(self, values: np.ndarray):
        """Sets the masks coefficients.

        @param value: Masks coefficients.
        @type value: np.ndarray
        @raise TypeError: If the masks is not a numpy array.
        @raise ValueError: If the masks is not 3D.
        """
        if not isinstance(values, np.ndarray):
            raise TypeError("Masks must be a numpy array.")
        if len(values.shape) != 3:
            raise ValueError("Masks must be 3D.")
        self._masks = values
