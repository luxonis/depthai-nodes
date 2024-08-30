from typing import List, Tuple, Union

import depthai as dai


class ImgDetectionWithKeypoints(dai.ImgDetection):
    """ImgDetectionWithKeypoints class for storing image detection with keypoints.

    Attributes
    ----------
    keypoints: List[Tuple[float, float]]
        Keypoints of the image detection.
    """

    def __init__(self):
        """Initializes the ImgDetectionWithKeypoints object."""
        dai.ImgDetection.__init__(self)  # TODO: change to super().__init__()?
        self._keypoints: List[Tuple[float, float]] = []

    @property
    def keypoints(self) -> List[Tuple[float, float]]:
        """Returns the keypoints.

        @return: List of keypoints.
        @rtype: List[Tuple[float, float]]
        """
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value: List[Tuple[Union[int, float], Union[int, float]]]):
        """Sets the keypoints.

        @param value: List of keypoints.
        @type value: List[Tuple[Union[int, float], Union[int, float]]]
        @raise TypeError: If the keypoints are not a list.
        @raise TypeError: If each keypoint is not a tuple of two floats or integers.
        """
        if not isinstance(value, list):
            raise TypeError("Keypoints must be a list")
        for item in value:
            if (
                not (isinstance(item, tuple) or isinstance(item, list))
                or len(item) != 2
                or not all(isinstance(i, (int, float)) for i in item)
            ):
                raise TypeError(
                    "Each keypoint must be a tuple of two floats or integers."
                )
        self._keypoints = [(float(x), float(y)) for x, y in value]


class ImgDetectionsWithKeypoints(dai.Buffer):
    """ImgDetectionsWithKeypoints class for storing image detections with keypoints.

    Attributes
    ----------
    detections: List[ImgDetectionWithKeypoints]
        Image detections with keypoints.
    """

    def __init__(self):
        """Initializes the ImgDetectionsWithKeypoints object."""
        dai.Buffer.__init__(self)  # TODO: change to super().__init__()?
        self._detections: List[ImgDetectionWithKeypoints] = []

    @property
    def detections(self) -> List[ImgDetectionWithKeypoints]:
        """Returns the image detections with keypoints.

        @return: List of image detections with keypoints.
        @rtype: List[ImgDetectionWithKeypoints]
        """
        return self._detections

    @detections.setter
    def detections(self, value: List[ImgDetectionWithKeypoints]):
        """Sets the image detections with keypoints.

        @param value: List of image detections with keypoints.
        @type value: List[ImgDetectionWithKeypoints]
        @raise TypeError: If the detections are not a list.
        @raise TypeError: If each detection is not an instance of
            ImgDetectionWithKeypoints.
        """
        if not isinstance(value, list):
            raise TypeError("Detections must be a list")
        for item in value:
            if not isinstance(item, ImgDetectionWithKeypoints):
                raise TypeError(
                    "Each detection must be an instance of ImgDetectionWithKeypoints"
                )
        self._detections = value
