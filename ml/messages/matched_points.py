import depthai as dai
from typing import List

class MatchedPoints(dai.Buffer):
    def __init__(self):
        super().__init__()
        self._reference_points: List[List[float]] = []
        self._target_points: List[List[float]] = []

    @property
    def reference_points(self) -> List[List[float]]:
        return self._reference_points

    @reference_points.setter
    def reference_points(self, value: List[List[float]]):
        if not isinstance(value, list):
            raise TypeError("reference_points must be a list.")
        for item in value:
            if not isinstance(item, list):
                raise TypeError("reference points should be List[List[float]].")
        for item in value:
            if len(item) != 2:
                raise ValueError("Each item in reference_points must be of length 2.")
        for item in value:
            if not all(isinstance(i, float) for i in item):
                raise TypeError("All items in reference_points must be of type float.")
        self._reference_points = value

    @property
    def target_points(self) -> List[List[float]]:
        return self._target_points
    
    @target_points.setter
    def target_points(self, value: List[List[float]]):
        if not isinstance(value, list):
            raise TypeError("target_points must be a list.")
        for item in value:
            if not isinstance(item, list):
                raise TypeError("target points should be List[List[float]].")
        for item in value:
            if len(item) != 2:
                raise ValueError("Each item in target_points must be of length 2.")
        for item in value:
            if not all(isinstance(i, float) for i in item):
                raise TypeError("All items in target_points must be of type float.")
        self._target_points = value
