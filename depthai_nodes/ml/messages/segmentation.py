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
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self):
        """Initializes the SegmentationMask object."""
        super().__init__()
        self._mask: NDArray[np.int16] = np.array([])
        self._transformation: dai.ImgTransformation = None

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

    @property
    def transformation(self) -> dai.ImgTransformation:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self._transformation

    @transformation.setter
    def transformation(self, value: dai.ImgTransformation):
        """Sets the Image Transformation object.

        @param value: The Image Transformation object.
        @type value: dai.ImgTransformation
        @raise TypeError: If value is not a dai.ImgTransformation object.
        """

        if value is not None:
            if not isinstance(value, dai.ImgTransformation):
                raise TypeError(
                    f"Transformation must be a dai.ImgTransformation object, instead got {type(value)}."
                )
        self._transformation = value
