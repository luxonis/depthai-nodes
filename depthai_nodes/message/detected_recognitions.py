import depthai as dai
from depthai_nodes import ImgDetectionsExtended


class DetectedRecognitions(dai.Buffer):
    def __init__(
        self, detections: dai.ImgDetections | ImgDetectionsExtended, nn_data: list[dai.NNData]
    ) -> None:
        super().__init__(0)
        self.img_detections: dai.ImgDetections = detections
        self.nn_data: list[dai.NNData] = nn_data

        self.setTimestampDevice(detections.getTimestampDevice())
        self.setTimestamp(detections.getTimestamp())
        self.setSequenceNum(detections.getSequenceNum())
