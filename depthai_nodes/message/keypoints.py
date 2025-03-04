from typing import List

import depthai as dai

from depthai_nodes import KEYPOINT_COLOR
from depthai_nodes.logging import get_logger


class Keypoint(dai.Buffer):
    """Keypoint class for storing a keypoint.

    Attributes
    ----------
    x: float
        X coordinate of the keypoint, relative to the input height.
    y: float
        Y coordinate of the keypoint, relative to the input width.
    z: Optional[float]
        Z coordinate of the keypoint.
    confidence: Optional[float]
        Confidence of the keypoint.
    """

    def __init__(self):
        """Initializes the Keypoint object."""
        super().__init__()
        self._x: float = None
        self._y: float = None
        self._z: float = 0.0
        self._confidence: float = -1.0
        self._logger = get_logger(__name__)

    @property
    def x(self) -> float:
        """Returns the X coordinate of the keypoint.

        @return: X coordinate of the keypoint.
        @rtype: float
        """
        return self._x

    @x.setter
    def x(self, value: float):
        """Sets the X coordinate of the keypoint.

        @param value: X coordinate of the keypoint.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise ValueError: If value is not between 0 and 1.
        """
        if not isinstance(value, float):
            raise TypeError("x must be a float.")
        if value < -0.1 or value > 1.1:
            raise ValueError("x must be between 0 and 1.")
        if not (0 <= value <= 1):
            value = max(0, min(1, value))
            self._logger.info("x value was clipped to [0, 1].")
        self._x = value

    @property
    def y(self) -> float:
        """Returns the Y coordinate of the keypoint.

        @return: Y coordinate of the keypoint.
        @rtype: float
        """
        return self._y

    @y.setter
    def y(self, value: float):
        """Sets the Y coordinate of the keypoint.

        @param value: Y coordinate of the keypoint.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise ValueError: If value is not between 0 and 1.
        """
        if not isinstance(value, float):
            raise TypeError("y must be a float.")
        if value < -0.1 or value > 1.1:
            raise ValueError("y must be between 0 and 1.")
        if not (0 <= value <= 1):
            value = max(0, min(1, value))
            self._logger.info("y value was clipped to [0, 1].")
        self._y = value

    @property
    def z(self) -> float:
        """Returns the Z coordinate of the keypoint.

        @return: Z coordinate of the keypoint.
        @rtype: float
        """
        return self._z

    @z.setter
    def z(self, value: float):
        """Sets the Z coordinate of the keypoint.

        @param value: Z coordinate of the keypoint.
        @type value: float
        @raise TypeError: If value is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("z must be a float.")
        self._z = value

    @property
    def confidence(self) -> float:
        """Returns the confidence of the keypoint.

        @return: Confidence of the keypoint.
        @rtype: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value: float):
        """Sets the confidence of the keypoint.

        @param value: Confidence of the keypoint.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise ValueError: If value is not between 0 and 1.
        """
        if not isinstance(value, float):
            raise TypeError("confidence must be a float.")
        if value < -0.1 or value > 1.1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not (0 <= value <= 1):
            value = max(0, min(1, value))
            self._logger.info("Confidence value was clipped to [0, 1].")
        self._confidence = value


class Keypoints(dai.Buffer):
    """Keypoints class for storing keypoints.

    Attributes
    ----------
    keypoints: List[Keypoint]
        List of Keypoint objects, each representing a keypoint.
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self):
        """Initializes the Keypoints object."""
        super().__init__()
        self._keypoints: List[Keypoint] = []
        self._transformation: dai.ImgTransformation = None

    @property
    def keypoints(self) -> List[Keypoint]:
        """Returns the keypoints.

        @return: List of keypoints.
        @rtype: List[Keypoint]
        """
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value: List[Keypoint]):
        """Sets the keypoints.

        @param value: List of keypoints.
        @type value: List[Keypoint]
        @raise TypeError: If value is not a list.
        @raise TypeError: If each each element is not of type Keypoint.
        """
        if not isinstance(value, list):
            raise TypeError("keypoints must be a list.")
        if not all(isinstance(item, Keypoint) for item in value):
            raise ValueError("keypoints must be a list of Keypoint objects.")
        self._keypoints = value

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
        """Creates a default visualization message for the keypoints."""
        img_annotations = dai.ImgAnnotations()
        annotation = dai.ImgAnnotation()
        keypoints = [dai.Point2f(keypoint.x, keypoint.y) for keypoint in self.keypoints]
        pointsAnnotation = dai.PointsAnnotation()
        pointsAnnotation.type = dai.PointsAnnotationType.POINTS
        pointsAnnotation.points = dai.VectorPoint2f(keypoints)
        pointsAnnotation.outlineColor = KEYPOINT_COLOR
        pointsAnnotation.fillColor = KEYPOINT_COLOR
        pointsAnnotation.thickness = 2
        annotation.points.append(pointsAnnotation)

        img_annotations.annotations.append(annotation)
        img_annotations.setTimestamp(self.getTimestamp())
        return img_annotations
