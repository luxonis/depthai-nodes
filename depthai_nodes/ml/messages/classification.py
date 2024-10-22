from typing import List

import depthai as dai
import numpy as np
from numpy.typing import NDArray


class Classifications(dai.Buffer):
    """Classification class for storing the classes and their respective scores.

    Attributes
    ----------
    classes : list[str]
        A list of classes.
    scores : NDArray[np.float32]
        Corresponding probability scores.
    """

    def __init__(self):
        """Initializes the Classifications object."""
        dai.Buffer.__init__(self)
        self._classes: List[str] = []
        self._scores: NDArray[np.float32] = np.array([])

    @property
    def classes(self) -> List:
        """Returns the list of classes.

        @return: List of classes.
        @rtype: List[str]
        """
        return self._classes

    @classes.setter
    def classes(self, value: List[str]):
        """Sets the classes.

        @param value: A list of class names.
        @type value: List[str]
        @raise TypeError: If value is not a list.
        @raise ValueError: If each element is not of type string.
        """
        if not isinstance(value, List):
            raise TypeError(f"Classes must be a list, instead got {type(value)}.")
        if not all(isinstance(class_name, str) for class_name in value):
            raise ValueError("Classes must be a list of strings.")
        self._classes = value

    @property
    def scores(self) -> NDArray:
        """Returns the list of scores.

        @return: List of scores.
        @rtype: NDArray[np.float32]
        """
        return self._scores

    @scores.setter
    def scores(self, value: NDArray[np.float32]):
        """Sets the scores.

        @param value: A list of scores.
        @type value: NDArray[np.float32]
        @raise TypeError: If value is not a numpy array.
        @raise ValueError: If value is not a 1D numpy array.
        @raise ValueError: If each element is not of type float.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Scores must be a np.ndarray, instead got {type(value)}.")
        if value.ndim != 1:
            raise ValueError("Scores must be a 1D a np.ndarray.")
        if value.dtype != np.float32:
            raise ValueError("Scores must be a np.ndarray of floats.")
        self._scores = value
