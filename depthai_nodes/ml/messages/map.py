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
    """

    def __init__(self):
        """Initializes the Map2D object."""
        super().__init__()
        self._map: NDArray[np.float32] = np.array([])
        self._width: int = None
        self._height: int = None

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

    def getVisualizationMessage(self) -> dai.ImgFrame:
        """Returns default visualization message for 2D maps in the form of a
        colormapped image."""
        img_frame = dai.ImgFrame()
        mask = self._map.copy()
        if np.any(mask < 1):
            mask = mask * 255
        mask = mask.astype(np.uint8)
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_PLASMA)

        img_frame.setTimestamp(self.getTimestamp())
        return img_frame.setCvFrame(colored_mask, dai.ImgFrame.Type.BGR888i)
