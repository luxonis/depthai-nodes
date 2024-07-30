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
        @raise TypeError: If the start point is not of type dai.Point2f.
        """
        if not isinstance(value, dai.Point2f):
            raise TypeError(
                f"start_point must be of type Point2f, instead got {type(value)}."
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
        @raise TypeError: If the end point is not of type dai.Point2f.
        """
        if not isinstance(value, dai.Point2f):
            raise TypeError(
                f"end_point must be of type Point2f, instead got {type(value)}."
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
        @raise TypeError: If the confidence is not of type float.
        """
        if not isinstance(value, float):
            raise TypeError(
                f"confidence must be of type float, instead got {type(value)}."
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
        @raise TypeError: If the lines are not a list.
        @raise TypeError: If each line is not of type Line.
        """
        if not isinstance(value, List):
            raise TypeError(
                f"lines must be of type List[Line], instead got {type(value)}."
            )
        for line in value:
            if not isinstance(line, Line):
                raise TypeError(
                    f"lines must be of type List[Line], instead got {type(value)}."
                )
        self._lines = value
