from typing import List

import depthai as dai


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
        @raise TypeError: If value is not of type float.
        """
        if not isinstance(value, float):
            raise TypeError(
                f"Confidence must be of type float, instead got {type(value)}."
            )
        self._confidence = value


class Lines(dai.Buffer):
    """Lines class for storing lines.

    Attributes
    ----------
    lines : List[Line]
        List of detected lines.
    """

    def __init__(self):
        """Initializes the Lines object."""
        super().__init__()
        self._lines: List[Line] = []

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
