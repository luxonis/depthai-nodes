from typing import List

import depthai as dai
import numpy as np
from numpy.typing import NDArray

from depthai_nodes.utils.constants import (
    TEXT_BACKGROUND_COLOR,
    TEXT_COLOR,
)


class Classifications(dai.Buffer):
    """Classification class for storing the classes and their respective scores.

    Attributes
    ----------
    classes : list[str]
        A list of classes.
    scores : NDArray[np.float32]
        Corresponding probability scores.
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self):
        """Initializes the Classifications object."""
        dai.Buffer.__init__(self)
        self._classes: List[str] = []
        self._scores: NDArray[np.float32] = np.array([])
        self._transformation: dai.ImgTransformation = None

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
        if value.size > 0 and value.dtype != np.float32:
            raise ValueError("Scores must be a np.ndarray of floats.")
        self._scores = value

    @property
    def top_class(self) -> str:
        """Returns the most probable class. Only works if classes are sorted by scores.

        @return: The top class.
        @rtype: str
        """
        return self._classes[0]

    @property
    def top_score(self) -> float:
        """Returns the probability of the most probable class. Only works if scores are
        sorted by descending order.

        @return: The top score.
        @rtype: float
        """
        return self._scores[0]

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

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        """Returns default visualization message for classification.

        The message adds the top five classes and their scores to the right side of the
        image.
        """
        img_annotations = dai.ImgAnnotations()
        annotation = dai.ImgAnnotation()

        for i in range(len(self._scores)):
            text = dai.TextAnnotation()
            text.position = dai.Point2f(1.05, 0.1 + i * 0.1)
            text.text = f"{self._classes[i]} {self._scores[i] * 100:.0f}%"
            text.fontSize = 15
            text.textColor = TEXT_COLOR
            text.backgroundColor = TEXT_BACKGROUND_COLOR
            annotation.texts.append(text)

        img_annotations.annotations.append(annotation)
        img_annotations.setTimestamp(self.getTimestamp())
        return img_annotations
