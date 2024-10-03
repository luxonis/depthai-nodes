from typing import List

import depthai as dai
import numpy as np

from .keypoints import Keypoints


class ImgDetectionExtended(dai.ImgDetection):
    """ImgDetectionExtended class for storing image detection with keypoints and masks.

    Attributes
    ----------
    keypoints: Keypoints
        Keypoints of the detection.
    masks: np.ndarray
        Segmentation Mask of the detection.
    angle: float
        Angle of the detection.
    """

    def __init__(self):
        """Initializes the ImgDetectionExtended object."""
        super().__init__()
        self._keypoints: Keypoints = []
        self._mask: np.ndarray = np.array([])
        self._angle: float = 0.0

    @property
    def keypoints(
        self,
    ) -> Keypoints:
        """Returns the keypoints.

        @return: List of keypoints.
        @rtype: Keypoints
        """
        return self._keypoints

    @keypoints.setter
    def keypoints(
        self,
        value: Keypoints,
    ):
        """Sets the keypoints.

        @param value: List of keypoints.
        @type value: Keypoints
        @raise TypeError: If the keypoints are not a Keypoints object.
        """
        if not isinstance(value, Keypoints):
            raise TypeError("Keypoints must be a Keypoints object")
        self._keypoints = value

    @property
    def mask(self) -> np.ndarray:
        """Returns the mask coefficients.

        @return: Mask coefficients.
        @rtype: np.ndarray
        """
        return self._mask

    @mask.setter
    def mask(self, value: np.ndarray):
        """Sets the mask coefficients.

        @param value: Mask coefficients.
        @type value: np.ndarray
        @raise TypeError: If the mask is not a numpy array.
        @raise ValueError: If the mask is not of shape (H/4, W/4).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Mask must be a numpy array")
        if len(value.shape) != 2:
            raise ValueError("Mask must be of shape (H/4, W/4)")
        self._mask = value


class ImgDetectionsExtended(dai.ImgDetections):
    """ImgDetectionsExtended class for storing image detections with keypoints.

    Attributes
    ----------
    detections: List[ImgDetectionExtended]
        Image detections with keypoints.
    """

    def __init__(self):
        """Initializes the ImgDetectionsExtended object."""
        super().__init__()
        self._detections: List[ImgDetectionExtended] = []

    @property
    def detections(self) -> List[ImgDetectionExtended]:
        """Returns the image detections with keypoints.

        @return: List of image detections with keypoints.
        @rtype: List[ImgDetectionExtended]
        """
        return self._detections

    @detections.setter
    def detections(self, value: List[ImgDetectionExtended]):
        """Sets the image detections with keypoints.

        @param value: List of image detections with keypoints.
        @type value: List[ImgDetectionExtended]
        @raise TypeError: If the detections are not a list.
        @raise TypeError: If each detection is not an instance of ImgDetectionExtended.
        """
        if not isinstance(value, list):
            raise TypeError("Detections must be a list")
        for item in value:
            if not isinstance(item, ImgDetectionExtended):
                raise TypeError(
                    "Each detection must be an instance of ImgDetectionExtended"
                )
        self._detections = value


class CornerDetections(dai.Buffer):
    """Detection Class for storing object detections in corner format.

    Attributes
    ----------
    detections: List[Keypoints]
        List of detections in keypoint format.

    labels: List[int]
        List of labels for each detection
    """

    def __init__(self):
        """Initializes the CornerDetections object."""
        dai.Buffer.__init__(self)
        self._detections: List[Keypoints] = []
        self._scores: List[float] = None
        self._labels: List[int] = None

    @property
    def detections(self) -> List[Keypoints]:
        """Returns the detections.

        @return: List of detections.
        @rtype: List[Keypoints]
        """
        return self._detections

    @detections.setter
    def detections(self, value: List[Keypoints]):
        """Sets the detections.

        @param value: List of detections.
        @type value: List[Keypoints]
        @raise TypeError: If the detections are not a list.
        @raise TypeError: If each detection is not an instance of Keypoints.
        """
        if not isinstance(value, list):
            raise TypeError("Detections must be a list")
        for item in value:
            if not isinstance(item, Keypoints):
                raise TypeError("Each detection must be an instance of Keypoints")
        self._detections = value

    @property
    def labels(self) -> List[int]:
        """Returns the labels.

        @return: List of labels.
        @rtype: List[int]
        """
        return self._labels

    @labels.setter
    def labels(self, value: List[int]):
        """Sets the labels.

        @param value: List of labels.
        @type value: List[int]
        @raise TypeError: If the labels are not a list.
        @raise TypeError: If each label is not an integer.
        """
        if not isinstance(value, list):
            raise TypeError("Labels must be a list")
        for item in value:
            if not isinstance(item, int):
                raise TypeError("Each label must be an integer")
        self._labels = value

    @property
    def scores(self) -> List[float]:
        """Returns the scores.

        @return: List of scores.
        @rtype: List[float]
        """
        return self._scores

    @scores.setter
    def scores(self, value: List[float]):
        """Sets the scores.

        @param value: List of scores.
        @type value: List[float]
        @raise TypeError: If the scores are not a list.
        @raise TypeError: If each score is not a float.
        """
        if not isinstance(value, list):
            raise TypeError("Scores must be a list")
        for item in value:
            if not isinstance(item, float):
                raise TypeError("Each score must be a float")
        self._scores = value
