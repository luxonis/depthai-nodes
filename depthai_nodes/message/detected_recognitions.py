from typing import List, Union

import depthai as dai

from .img_detections import ImgDetectionsExtended


class DetectedRecognitions(dai.Buffer):
    def __init__(
        self,
        detections: Union[dai.ImgDetections, ImgDetectionsExtended],
        nn_data: List[dai.NNData],
    ) -> None:
        super().__init__(0)
        self.img_detections: Union[
            dai.ImgDetections, ImgDetectionsExtended
        ] = detections
        self.nn_data: list[dai.NNData] = nn_data

        self.setTimestampDevice(detections.getTimestampDevice())
        self.setTimestamp(detections.getTimestamp())
        self.setSequenceNum(detections.getSequenceNum())
