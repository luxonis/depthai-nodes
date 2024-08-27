
import numpy as np
from typing import List, Tuple, Union

import depthai as dai


class ImgDetectionWithAdditionalOutput(dai.ImgDetection):
    """ImgDetectionWithAdditionalOutput class for storing image detection with keypoints and masks.

    Attributes
    ----------
    keypoints: List[Tuple[float, float]]
        Keypoints of the image detection.
    masks: np.ndarray
        Mask of the image segmentation.
    """

    def __init__(self):
        """Initializes the ImgDetectionWithAdditionalOutput object."""
        dai.ImgDetection.__init__(self)  # TODO: change to super().__init__()?
        self._keypoints: List[Tuple[float, float]] = []
        self._mask: np.ndarray = np.array([])

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
                    "Each keypoint must be a tuple of two floats or integers"
                )
        self._keypoints = [(float(x), float(y)) for x, y in value]

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

class ImgDetectionsWithAdditionalOutput(dai.Buffer):
    """ImgDetectionsWithAdditionalOutput class for storing image detections with keypoints.

    Attributes
    ----------
    detections: List[ImgDetectionWithAdditionalOutput]
        Image detections with keypoints.
    """

    def __init__(self):
        """Initializes the ImgDetectionsWithAdditionalOutput object."""
        dai.Buffer.__init__(self)  # TODO: change to super().__init__()?
        self._detections: List[ImgDetectionWithAdditionalOutput] = []

    @property
    def detections(self) -> List[ImgDetectionWithAdditionalOutput]:
        """Returns the image detections with keypoints.

        @return: List of image detections with keypoints.
        @rtype: List[ImgDetectionWithAdditionalOutput]
        """
        return self._detections

    @detections.setter
    def detections(self, value: List[ImgDetectionWithAdditionalOutput]):
        """Sets the image detections with keypoints.

        @param value: List of image detections with keypoints.
        @type value: List[ImgDetectionWithAdditionalOutput]
        @raise TypeError: If the detections are not a list.
        @raise TypeError: If each detection is not an instance of
            ImgDetectionWithAdditionalOutput.
        """
        if not isinstance(value, list):
            raise TypeError("Detections must be a list")
        for item in value:
            if not isinstance(item, ImgDetectionWithAdditionalOutput):
                raise TypeError(
                    "Each detection must be an instance of ImgDetectionWithAdditionalOutput"
                )
        self._detections = value
