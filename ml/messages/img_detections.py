import depthai as dai
from typing import List

class ImgDetectionWithKeypoints(dai.ImgDetection):
    def __init__(self):
        dai.ImgDetection.__init__(self)
        self.keypoints = [] # TODO: how to enforce type checking for keypoints? e.g. this currently accept strings

class ImgDetectionsWithKeypoints(dai.Buffer):
    def __init__(self):
        dai.Buffer.__init__(self)
        self.detections: List[ImgDetectionWithKeypoints] = [] # TODO: how to enforce type checking for ImgDetectionWithKeypoints? e.g. this currently accept ImgDetection