import copy
from typing import List, Optional

import depthai as dai

from depthai_nodes import PRIMARY_COLOR
from depthai_nodes.logging import get_logger
from depthai_nodes.utils import AnnotationHelper

from .utils import (
    copy_message,
)


class Line(dai.Buffer):
    """Line class for storing a line.

    Attributes
    ----------
    start_point : dai.Point2f
        Start point of the line with x and y coordinate.
    end_point : dai.Point2f
        End point of the line with x and y coordinate.
    confidence : float
        Confidence of the line.
    """

    def __init__(self):
        """Initializes the Line object."""
        super().__init__()
        self._start_point: dai.Point2f = None
        self._end_point: dai.Point2f = None
        self._confidence: float = None
        self._logger = get_logger(__name__)

    def copy(self):
        """Creates a new instance of the Line class and copies the attributes.

        @return: A new instance of the Line class.
        @rtype: Line
        """
        new_obj = Line()
        new_obj.start_point = copy_message(self.start_point)
        new_obj.end_point = copy_message(self.end_point)
        new_obj.confidence = copy.deepcopy(self.confidence)
        return new_obj

    @property
    def start_point(self) -> dai.Point2f:
        """Returns the start point of the line.

        @return: Start point of the line.
        @rtype: dai.Point2f
        """
        return self._start_point

    @start_point.setter
    def start_point(self, value: dai.Point2f):
        """Sets the start point of the line.

        @param value: Start point of the line.
        @type value: dai.Point2f
        @raise TypeError: If value is not of type dai.Point2f.
        """
        if not isinstance(value, dai.Point2f):
            raise TypeError(
                f"Start Point must be of type Point2f, instead got {type(value)}."
            )
        self._start_point = value

    @property
    def end_point(self) -> dai.Point2f:
        """Returns the end point of the line.

        @return: End point of the line.
        @rtype: dai.Point2f
        """
        return self._end_point

    @end_point.setter
    def end_point(self, value: dai.Point2f):
        """Sets the end point of the line.

        @param value: End point of the line.
        @type value: dai.Point2f
        @raise TypeError: If value is not of type dai.Point2f.
        """
        if not isinstance(value, dai.Point2f):
            raise TypeError(
                f"End Point must be of type Point2f, instead got {type(value)}."
            )
        self._end_point = value

    @property
    def confidence(self) -> float:
        """Returns the confidence of the line.

        @return: Confidence of the line.
        @rtype: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value: float):
        """Sets the confidence of the line.

        @param value: Confidence of the line.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise ValueError: If value is not between 0 and 1.
        """
        if not isinstance(value, float):
            raise TypeError("Confidence must be a float.")
        if value < -0.1 or value > 1.1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not (0 <= value <= 1):
            value = float(max(0.0, min(1.0, value)))
            self._logger.info("Confidence value was clipped to [0, 1].")

        self._confidence = value


class Lines(dai.Buffer):
    """Lines class for storing lines.

    Attributes
    ----------
    lines : List[Line]
        List of detected lines.
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self):
        """Initializes the Lines object."""
        super().__init__()
        self._lines: List[Line] = []
        self._transformation: Optional[dai.ImgTransformation] = None

    def copy(self):
        """Creates a new instance of the Lines class and copies the attributes.

        @return: A new instance of the Lines class.
        @rtype: Lines
        """
        new_obj = Lines()
        new_obj.lines = [line.copy() for line in self.lines]
        new_obj.setSequenceNum(self.getSequenceNum())
        new_obj.setTimestamp(self.getTimestamp())
        new_obj.setTimestampDevice(self.getTimestampDevice())
        new_obj.setTransformation(self.transformation)
        return new_obj

    @property
    def lines(self) -> List[Line]:
        """Returns the lines.

        @return: List of lines.
        @rtype: List[Line]
        """
        return self._lines

    @lines.setter
    def lines(self, value: List[Line]):
        """Sets the lines.

        @param value: List of lines.
        @type value: List[Line]
        @raise TypeError: If value is not a list.
        @raise TypeError: If each element is not of type Line.
        """
        if not isinstance(value, List):
            raise TypeError(f"lines must be a list, instead got {type(value)}.")
        if not all(isinstance(item, Line) for item in value):
            raise ValueError("Lines must be a list of Line objects.")
        self._lines = value

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
        return self._transformation

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        """Returns default visualization message for lines.

        The message adds lines to the image.
        """
        annotation_helper = AnnotationHelper()
        for line in self.lines:
            annotation_helper.draw_line(
                pt1=line.start_point,
                pt2=line.end_point,
                color=PRIMARY_COLOR,
                thickness=2.0,
            )
        return annotation_helper.build(
            timestamp=self.getTimestamp(), sequence_num=self.getSequenceNum()
        )
