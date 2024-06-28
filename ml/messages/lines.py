import depthai as dai
from typing import List

class Line(dai.Buffer):
    def __init__(self):
        super().__init__()
        self._start_point: dai.Point2f = None
        self._end_point: dai.Point2f = None
        self._confidence: float = None

    @property
    def start_point(self) -> dai.Point2f:
        return self._start_point

    @start_point.setter
    def start_point(self, value: dai.Point2f):
        if not isinstance(value, dai.Point2f):
            raise TypeError(f"start_point must be of type Point2f, instead got {type(value)}.")
        self._start_point = value

    @property
    def end_point(self) -> dai.Point2f:
        return self._end_point
    
    @end_point.setter
    def end_point(self, value: dai.Point2f):
        if not isinstance(value, dai.Point2f):
            raise TypeError(f"end_point must be of type Point2f, instead got {type(value)}.")
        self._end_point = value

    @property
    def confidence(self) -> float:
        return self._confidence
    
    @confidence.setter
    def confidence(self, value: float):
        if not isinstance(value, float):
            raise TypeError(f"confidence must be of type float, instead got {type(value)}.")
        self._confidence = value


class Lines(dai.Buffer):
    def __init__(self):
        super().__init__()
        self._lines: List[Line] = []

    @property
    def lines(self) -> List[Line]:
        return self._lines

    @lines.setter
    def lines(self, value: List[Line]):
        if not isinstance(value, List):
            raise TypeError(f"lines must be of type List[Line], instead got {type(value)}.")
        for line in value:
            if not isinstance(line, Line):
                raise TypeError(f"lines must be of type List[Line], instead got {type(value)}.")
        self._lines = value