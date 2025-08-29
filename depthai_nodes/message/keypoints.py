import copy
from typing import List, Optional, Tuple

import depthai as dai

from depthai_nodes import KEYPOINT_COLOR, PRIMARY_COLOR
from depthai_nodes.logging import get_logger
from depthai_nodes.utils import AnnotationHelper


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
    label_name: Optional[str]
        Label name of the keypoint.
    """

    def __init__(self):
        """Initializes the Keypoint object."""
        super().__init__()
        self._x: float = None
        self._y: float = None
        self._z: float = 0.0
        self._confidence: float = -1.0
        self._label_name: str = None
        self._logger = get_logger(__name__)

    def copy(self):
        """Creates a new instance of the Keypoint class and copies the attributes.

        @return: A new instance of the Keypoint class.
        @rtype: Keypoint
        """
        new_obj = Keypoint()
        new_obj.x = copy.deepcopy(self.x)
        new_obj.y = copy.deepcopy(self.y)
        new_obj.z = copy.deepcopy(self.z)
        new_obj.confidence = copy.deepcopy(self.confidence)
        new_obj.label_name = copy.deepcopy(self.label_name)
        return new_obj

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
        if not (0 <= value <= 1):
            value = float(max(0.0, min(1.0, value)))
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
        if not (0 <= value <= 1):
            value = float(max(0.0, min(1.0, value)))
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
        if (value < -0.1 or value > 1.1) and value != -1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        if not (0 <= value <= 1):
            value = float(max(0.0, min(1.0, value)))
            self._logger.info("Confidence value was clipped to [0, 1].")
        self._confidence = value

    @property
    def label_name(self) -> str:
        """Returns the label name of the keypoint.

        @return: Label name of the keypoint.
        @rtype: str
        """
        return self._label_name

    @label_name.setter
    def label_name(self, value: str):
        """Sets the label name of the keypoint.

        @param value: Label name of the keypoint.
        @type value: str
        """
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("label_name must be a string.")
        self._label_name = value


class Keypoints(dai.Buffer):
    """Keypoints class for storing keypoints.

    Attributes
    ----------
    keypoints: List[Keypoint]
        List of Keypoint objects, each representing a keypoint.
    edges: List[Tuple[int, int]]
        List of edges, each representing a connection between two keypoints. NOTE: If you create a Keypoints message with a `create_keypoints_message` function, the edges will be filtered to only include the edges between the keypoints that are present in the filtered keypoints. This is done to ensure that the edges are only drawn between the keypoints that are present in the filtered keypoints. You can always access the full set of edges in the model's NN archive.
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self):
        """Initializes the Keypoints object."""
        super().__init__()
        self._keypoints: List[Keypoint] = []
        self._edges: List[Tuple[int, int]] = []
        self._transformation: Optional[dai.ImgTransformation] = None

    def copy(self):
        """Creates a new instance of the Keypoints class and copies the attributes.

        @return: A new instance of the Keypoints class.
        @rtype: Keypoints
        """
        new_obj = Keypoints()
        new_obj.keypoints = [keypoint.copy() for keypoint in self.keypoints]
        new_obj.edges = copy.deepcopy(self.edges)
        new_obj.setSequenceNum(self.getSequenceNum())
        new_obj.setTimestamp(self.getTimestamp())
        new_obj.setTimestampDevice(self.getTimestampDevice())
        new_obj.setTransformation(self.transformation)
        return new_obj

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
    def edges(self) -> List[Tuple[int, int]]:
        """Returns the edges.

        @return: List of edges.
        @rtype: List[Tuple[int, int]]
        """
        return self._edges

    @edges.setter
    def edges(self, value: List[Tuple[int, int]]):
        """Sets the edges.

        @param value: List of edges.
        @type value: List[Tuple[int, int]]
        @raise TypeError: If value is not a list.
        @raise TypeError: If each each element is not of type Tuple[int, int].
        """
        if not isinstance(value, list):
            raise TypeError("edges must be a list.")
        if not all(
            (isinstance(item, tuple) or isinstance(item, list))
            and len(item) == 2
            and all(isinstance(i, int) for i in item)
            for item in value
        ):
            raise TypeError("edges must be a list of tuples or lists of integers.")
        self._edges = value

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

    def setTransformation(self, transformation: Optional[dai.ImgTransformation]):
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

    def getPoints2f(self) -> dai.VectorPoint2f:
        """Returns the keypoints in the form of a dai.VectorPoint2f object."""

        return dai.VectorPoint2f(
            [dai.Point2f(keypoint.x, keypoint.y) for keypoint in self.keypoints]
        )

    def getPoints3f(self) -> List[dai.Point3f]:
        """Returns the keypoints in the form of a list of dai.Point3f objects."""

        return [
            dai.Point3f(keypoint.x, keypoint.y, keypoint.z)
            for keypoint in self.keypoints
        ]

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        """Creates a default visualization message for the keypoints."""
        annotation_helper = AnnotationHelper()
        annotation_helper.draw_points(
            points=self.getPoints2f(), color=KEYPOINT_COLOR, thickness=1
        )
        for edge in self.edges:
            pt1_ix, pt2_ix = edge
            pt1 = self.keypoints[pt1_ix]
            pt2 = self.keypoints[pt2_ix]
            annotation_helper.draw_line(
                pt1=(pt1.x, pt1.y),
                pt2=(pt2.x, pt2.y),
                color=PRIMARY_COLOR,
                thickness=1,
            )
        return annotation_helper.build(
            timestamp=self.getTimestamp(), sequence_num=self.getSequenceNum()
        )
