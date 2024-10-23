from typing import List

import depthai as dai
import numpy as np

from .keypoints import Keypoint
from .segmentation import SegmentationMask


class ImgDetectionExtended(dai.Buffer):
    """A class for storing image detections in (x_center, y_center, width, height)
    format with additional angle and keypoints.

    Attributes
    ----------
    x_center: float
        The X coordinate of the center of the bounding box, relative to the input width.
    y_center: float
        The Y coordinate of the center of the bounding box, relative to the input height.
    width: float
        The width of the bounding box, relative to the input width.
    height: float
        The height of the bounding box, relative to the input height.
    angle: float
        The angle of the bounding box expressed in degrees.
    confidence: float
        Confidence of the detection.
    label: int
        Label of the detection.
    keypoints: List[Keypoint]
        Keypoints of the detection.
    """

    def __init__(self):
        """Initializes the ImgDetectionExtended object."""
        super().__init__()
        self._x_center: float
        self._y_center: float
        self._width: float
        self._height: float

        self._angle: float = 0.0
        self._confidence: float = -1.0
        self._label: int = -1
        self._keypoints: List[Keypoint] = []

    @property
    def x_center(self) -> float:
        """Returns the X coordinate of the center of the bounding box.

        @return: X coordinate of the center of the bounding box.
        @rtype: float
        """
        return self._x_center

    @x_center.setter
    def x_center(self, value: float):
        """Sets the X coordinate of the center of the bounding box.

        @param value: X coordinate of the center of the bounding box.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise ValueError: If value is not between 0 and 1.
        """
        if not isinstance(value, float):
            raise TypeError("X center must be a float.")
        if value <= 0 or value >= 1:
            raise ValueError("X center must be between 0 and 1.")
        self._x_center = value

    @property
    def y_center(self) -> float:
        """Returns the Y coordinate of the center of the bounding box.

        @return: Y coordinate of the center of the bounding box.
        @rtype: float
        """
        return self._y_center

    @y_center.setter
    def y_center(self, value: float):
        """Sets the Y coordinate of the center of the bounding box.

        @param value: Y coordinate of the center of the bounding box.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise ValueError: If value is not between 0 and 1.
        """
        if not isinstance(value, float):
            raise TypeError("Y center must be a float.")
        if value <= 0 or value >= 1:
            raise ValueError("Y center must be between 0 and 1.")
        self._y_center = value

    @property
    def width(self) -> float:
        """Returns the width of the bounding box.

        @return: Width of the bounding box.
        @rtype: float
        """
        return self._width

    @width.setter
    def width(self, value: float):
        """Sets the width of the bounding box.

        @param value: Width of the bounding box.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise ValueError: If value is not between 0 and 1.
        """
        if not isinstance(value, float):
            raise TypeError("Width must be a float.")
        if value <= 0 or value >= 1:
            raise ValueError("Width must be between 0 and 1.")

        self._width = value

    @property
    def height(self) -> float:
        """Returns the height of the bounding box.

        @return: Height of the bounding box.
        @rtype: float
        """
        return self._height

    @height.setter
    def height(self, value: float):
        """Sets the height of the bounding box.

        @param value: Height of the bounding box.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise ValueError: If value is not between 0 and 1.
        """
        if not isinstance(value, float):
            raise TypeError("Height must be a float.")
        if value <= 0 or value >= 1:
            raise ValueError("Height must be between 0 and 1.")
        self._height = value

    @property
    def angle(self) -> float:
        """Returns the angle of the bounding box.

        @return: Angle of the bounding box.
        @rtype: float
        """
        return self._angle

    @angle.setter
    def angle(self, value: float):
        """Sets the angle of the bounding box.

        @param value: Angle of the bounding box.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise TypeError: If value is not between -360 and 360.
        """
        if not isinstance(value, float):
            raise TypeError("Angle must be a float.")
        if value < -360 or value > 360:
            raise TypeError("Angle must be between -360 and 360 degrees.")
        self._angle = value

    @property
    def confidence(self) -> float:
        """Returns the confidence of the detection.

        @return: Confidence of the detection.
        @rtype: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value: float):
        """Sets the confidence of the detection.

        @param value: Confidence of the detection.
        @type value: float
        @raise TypeError: If value is not a float.
        @raise ValueError: If value is not between 0 and 1.
        """
        if not isinstance(value, float):
            raise TypeError("Confidence must be a float.")
        if value <= 0 or value >= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        self._confidence = value

    @property
    def label(self) -> int:
        """Returns the label of the detection.

        @return: Label of the detection.
        @rtype: int
        """
        return self._label

    @label.setter
    def label(self, value: int):
        """Sets the label of the detection.

        @param value: Label of the detection.
        @type value: int
        @raise TypeError: If value is not an integer.
        """
        if not isinstance(value, int):
            raise TypeError("Label must be an integer.")
        self._label = value

    @property
    def keypoints(
        self,
    ) -> List[Keypoint]:
        """Returns the keypoints.

        @return: List of keypoints.
        @rtype: Keypoints
        """
        return self._keypoints

    @keypoints.setter
    def keypoints(
        self,
        value: List[Keypoint],
    ) -> None:
        """Sets the keypoints.

        @param value: List of keypoints.
        @type value: List[Keypoint]
        @raise TypeError: If value is not a list.
        @raise TypeError: If each element is not of type Keypoint.
        """
        if not isinstance(value, list):
            raise ValueError("Keypoints must be a list")
        if not all(isinstance(item, Keypoint) for item in value):
            raise ValueError("Keypoints must be a list of Keypoint objects.")
        self._keypoints = value


class ImgDetectionsExtended(dai.Buffer):
    """ImgDetectionsExtended class for storing image detections with keypoints.

    Attributes
    ----------
    detections: List[ImgDetectionExtended]
        Image detections with keypoints.
    masks: np.ndarray
        The segmentation masks of the image. All masks are stored in a single numpy array.
    """

    def __init__(self) -> None:
        """Initializes the ImgDetectionsExtended object.

        Attributes
        ----------
        detections: List[ImgDetectionExtended]
            Image detections with keypoints.
        masks: SegmentationMask
            The segmentation masks of the image stored in a single numpy array.
        """
        super().__init__()
        self._detections: List[ImgDetectionExtended] = []
        self._masks: SegmentationMask = np.array([])

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
        @raise TypeError: If value is not a list.
        @raise TypeError: If each element is not of type ImgDetectionExtended.
        """
        if not isinstance(value, list):
            raise TypeError("Detections must be a list.")
        if not all(isinstance(detection, ImgDetectionExtended) for detection in value):
            raise TypeError(
                "Detections must be a list of ImgDetectionExtended objects."
            )
        self._detections = value

    @property
    def masks(self) -> SegmentationMask:
        """Returns the segmentation masks stored in a single numpy array.

        @return: Segmentation masks.
        @rtype: SegmentationMask
        """
        return self._masks

    @masks.setter
    def masks(self, value: SegmentationMask):
        """Sets the masks of the image.

        @param masks: Mask coefficients.
        @type value: SegmentationMask
        @raise TypeError: If value is not of type SegmentationMask.
        """
        if not isinstance(value, SegmentationMask):
            raise TypeError("Mask must be a SegmentationMask object")
        self._masks = value
