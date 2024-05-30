import depthai as dai
from typing import List

class HandKeypoints(dai.Buffer):
    def __init__(self):
        dai.Buffer.__init__(self)
        self.confidence: float = 0.0
        self.handdedness: float = 0.0
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
