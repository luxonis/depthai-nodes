from typing import List

import depthai as dai


class Keypoints(dai.Buffer):
    def __init__(self):
        super().__init__()
        self._keypoints: List[dai.Point3f] = []

    @property
    def keypoints(self) -> List[dai.Point3f]:
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value: List[dai.Point3f]):
        if not isinstance(value, list):
            raise TypeError("keypoints must be a list.")
        for item in value:
            if not isinstance(item, dai.Point3f):
                raise TypeError("All items in keypoints must be of type dai.Point3f.")
        self._keypoints = value


class HandKeypoints(Keypoints):
    def __init__(self):
        Keypoints.__init__(self)
        self._confidence: float = 0.0
        self._handdedness: float = 0.0

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, value: float):
        if not isinstance(value, float):
            raise TypeError("confidence must be a float.")
        self._confidence = value

    @property
    def handdedness(self) -> float:
        return self._handdedness

    @handdedness.setter
    def handdedness(self, value: float):
        if not isinstance(value, float):
            raise TypeError("handdedness must be a float.")
        self._handdedness = value
