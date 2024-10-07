from typing import List, Tuple, Union

import depthai as dai
import numpy as np

from .keypoints import Keypoints


class ImgDetectionExtended(dai.ImgDetection):
    """ImgDetectionExtended class for storing image detection with keypoints and masks.

    Attributes
    ----------
    keypoints: Union[List[Tuple[float, float]], List[Tuple[float, float, float]]]
        Keypoints of the image detection.
    masks: np.ndarray
        Mask of the image segmentation.
    """

    def __init__(self):
        """Initializes the ImgDetectionExtended object."""
        dai.ImgDetection.__init__(self)  # TODO: change to super().__init__()?
        self._keypoints: Union[
            List[Tuple[float, float]], List[Tuple[float, float, float]]
        ] = []
        self._mask: np.ndarray = np.array([])

    @property
    def keypoints(
        self,
    ) -> Union[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
        """Returns the keypoints.

        @return: List of keypoints.
        @rtype: Union[List[Tuple[float, float]], List[Tuple[float, float, float]]]
        """
        return self._keypoints

    @keypoints.setter
    def keypoints(
        self,
        value: Union[
            List[Tuple[Union[int, float], Union[int, float]]],
            List[Tuple[Union[int, float], Union[int, float], Union[int, float]]],
        ],
    ):
        """Sets the keypoints.

        @param value: List of keypoints.
        @type value: Union[List[Tuple[Union[int, float], Union[int, float]]],
            List[Tuple[Union[int, float], Union[int, float], Union[int, float]]]]
        @raise TypeError: If the keypoints are not a list.
        @raise TypeError: If each keypoint is not a tuple of two or three floats or
            integers.
        """
        if not isinstance(value, list):
            raise TypeError("Keypoints must be a list")
        dim = len(value[0]) if value else 0
        for item in value:
            if (
                not (isinstance(item, tuple) or isinstance(item, list))
                or (len(item) != 2 and len(item) != 3)
                or not all(isinstance(i, (int, float)) for i in item)
            ):
                raise TypeError(
                    "Each keypoint must be a tuple of two or three floats or integers."
                )
            if len(item) != dim:
                raise ValueError(
                    "All keypoints must be of the same dimension e.g. [x, y] or [x, y, z], got mixed inner dimensions."
                )
        keypoints = []
        for items in value:
            keypoints.append(tuple(float(v) for v in items))
        self._keypoints = keypoints

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


class ImgDetectionsExtended(dai.Buffer):
    """ImgDetectionsExtended class for storing image detections with keypoints.

    Attributes
    ----------
    detections: List[ImgDetectionExtended]
        Image detections with keypoints.
    """

    def __init__(self):
        """Initializes the ImgDetectionsExtended object."""
        dai.Buffer.__init__(self)  # TODO: change to super().__init__()?
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
