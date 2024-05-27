import depthai as dai

class ImgDetectionsWithKeypoints(dai.ImgDetections):
    def __init__(self):
        dai.ImgDetections.__init__(self)
        self.keypoints = []