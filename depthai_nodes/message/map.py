import copy
from typing import Optional

import cv2
import depthai as dai
import numpy as np
from numpy.typing import NDArray


class Map2D(dai.Buffer):
    """Map2D class for storing a 2D map of floats.

    Attributes
    ----------
    map : NDArray[np.float32]
        2D map.
    width : int
        2D Map width.
    height : int
        2D Map height.
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self):
        """Initializes the Map2D object."""
        super().__init__()
        self._map: NDArray[np.float32] = np.array([])
        self._width: int = None
        self._height: int = None
        self._transformation: Optional[dai.ImgTransformation] = None

    def copy(self):
        """Creates a new instance of the Map2D class and copies the attributes.

        @return: A new instance of the Map2D class.
        @rtype: Map2D
        """
        new_obj = Map2D()
        new_obj.map = copy.deepcopy(self._map)
        new_obj.transformation = self._transformation
        return new_obj

    @property
    def map(self) -> NDArray[np.float32]:
        """Returns the 2D map.

        @return: 2D map.
        @rtype: NDArray[np.float32]
        """
        return self._map

    @map.setter
    def map(self, value: np.ndarray):
        """Sets the 2D map.

        @param value: 2D map.
        @type value: NDArray[np.float32]
        @raise TypeError: If value is not a numpy array.
        @raise ValueError: If value is not a 2D numpy array.
        @raise ValueError: If each element is not of type float.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"2D map must be of type np.ndarray, instead got {type(value)}."
            )
        if value.ndim != 2:
            raise ValueError("2D map must be a 2D array")
        if value.dtype != np.float32:
            raise ValueError("2D map must be an array of floats")
        self._map = value
        self._width = value.shape[1]
        self._height = value.shape[0]

    @property
    def width(self) -> int:
        """Returns the 2D map width.

        @return: 2D map width.
        @rtype: int
        """
        return self._width

    @property
    def height(self) -> int:
        """Returns the 2D map height.

        @return: 2D map height.
        @rtype: int
        """
        return self._height

    @property
    def transformation(self) -> Optional[dai.ImgTransformation]:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self._transformation

    @transformation.setter
    def transformation(self, value: Optional[dai.ImgTransformation]):
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

    def setTransformation(self, transformation: dai.ImgTransformation):
        """Sets the Image Transformation object.

        @param transformation: The Image Transformation object.
        @type transformation: dai.ImgTransformation
        @raise TypeError: If value is not a dai.ImgTransformation object.
        """
        self.transformation = transformation

    def getTransformation(self) -> Optional[dai.ImgTransformation]:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self.transformation

    def getVisualizationMessage(self) -> dai.ImgFrame:
        """Returns default visualization message for 2D maps in the form of a
        colormapped image."""
        img_frame = dai.ImgFrame()
        img_frame.setTimestamp(self.getTimestamp())
        img_frame.setTimestampDevice(self.getTimestampDevice())
        img_frame.setSequenceNum(self.getSequenceNum())
        if self.transformation is not None:
            img_frame.setTransformation(self.transformation)
        mask = self._map.copy()
        if np.any(mask < 1):
            mask = mask * 255
        mask = mask.astype(np.uint8)

        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_PLASMA)
        return img_frame.setCvFrame(colored_mask, dai.ImgFrame.Type.BGR888i)
