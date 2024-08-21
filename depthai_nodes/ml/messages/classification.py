from typing import List

import depthai as dai


class Classifications(dai.Buffer):
    """Classification class for storing the class names and their respective scores.

    Attributes
    ----------
    classes : list[str]
        A list of classes.
    scores : list[float]
        A list of corresponding probability scores.
    """

    def __init__(self):
        """Initializes the Classifications object and sets the classes and scores to
        empty lists."""
        dai.Buffer.__init__(self)
        self._classes: List[str] = []
        self._scores: List[float] = []

    @property
    def classes(self) -> List:
        """Returns the list of classes."""
        return self._classes

    @property
    def scores(self) -> List:
        """Returns the list of scores."""
        return self._scores

    @classes.setter
    def classes(self, class_names: List[str]):
        """Sets the list of classes.

        @param classes: A list of class names.
        """
        self._classes = class_names

    @scores.setter
    def scores(self, scores: List[float]):
        """Sets the list of scores.

        @param scores: A list of scores.
        """
        self._scores = scores
