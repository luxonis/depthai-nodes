import depthai as dai
from typing import List

class HandLandmarks(dai.Buffer):
    def __init__(self):
        dai.Buffer.__init__(self)
        self.confidence: float = 0.0
        self.handdedness: float = 0.0
        self.landmarks: List[dai.Point3f] = []