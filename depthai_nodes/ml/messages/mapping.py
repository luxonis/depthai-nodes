import depthai as dai
import numpy as np


class Map2D(dai.Buffer):
    def __init__(self):
        super().__init__()
        self._map: np.ndarray = None
        self._width: int = None
        self._height: int = None

    @property
    def map(self) -> np.ndarray:
        return self._map

    @map.setter
    def map(self, map: np.ndarray):
        if not isinstance(map, np.ndarray):
            raise TypeError(f"map must be of type np.ndarray, instead got {type(map)}.")
        if not len(map.shape) == 2:
            raise ValueError("array must be a 2D array")
        if map.dtype != np.float32:
            raise ValueError("array must be an array of floats")
        self._map = map
        self._width = map.shape[1]
        self._height = map.shape[0]
