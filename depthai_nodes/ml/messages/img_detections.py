from typing import List, Tuple, Union

import depthai as dai


class ImgDetectionWithKeypoints(dai.ImgDetection):
    def __init__(self):
        dai.ImgDetection.__init__(self)  # TODO: change to super().__init__()?
        self._keypoints: List[Tuple[float, float]] = []

    @property
    def keypoints(self) -> List[Tuple[float, float]]:
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value: List[Tuple[Union[int, float], Union[int, float]]]):
        if not isinstance(value, list):
            raise TypeError("Keypoints must be a list")
        for item in value:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(i, (int, float)) for i in item)
            ):
                raise TypeError(
                    "Each keypoint must be a tuple of two floats or integers"
                )
        self._keypoints = [(float(x), float(y)) for x, y in value]


class ImgDetectionsWithKeypoints(dai.Buffer):
    def __init__(self):
        dai.Buffer.__init__(self)  # TODO: change to super().__init__()?
        self._detections: List[ImgDetectionWithKeypoints] = []

    @property
    def detections(self) -> List[ImgDetectionWithKeypoints]:
        return self._detections

    @detections.setter
    def detections(self, value: List[ImgDetectionWithKeypoints]):
        if not isinstance(value, list):
            raise TypeError("Detections must be a list")
        for item in value:
            if not isinstance(item, ImgDetectionWithKeypoints):
                raise TypeError(
                    "Each detection must be an instance of ImgDetectionWithKeypoints"
                )
        self._detections = value
