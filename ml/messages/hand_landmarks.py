import depthai as dai
from typing import List

class HandLandmarksDescriptor:
    def __init__(self):
        self.name = 'landmarks'
        self.expected_type = dai.Point3f

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
            if not isinstance(value, list):
                raise TypeError(f"{self.name} must be a list")
            for item in value:
                if not isinstance(item, self.expected_type):
                    raise TypeError(f"All items in {self.name} must be of type {self.expected_type}")
            instance.__dict__[self.name] = value

class HandLandmarks(dai.Buffer):
    landmarks = HandLandmarksDescriptor()
    def __init__(self):
        dai.Buffer.__init__(self)
        self.confidence: float = 0.0
        self.handdedness: float = 0.0
        self.landmarks: List[dai.Point3f] = []