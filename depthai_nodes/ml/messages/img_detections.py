from typing import List

import depthai as dai
import numpy as np

from .keypoints import Keypoint


class ImgDetectionExtended(dai.Buffer):
    """A class for storing image detections in (x_center, y_center, width, height)
    format with additional angle and keypoints.

    Attributes
    ----------
    x_center: float
        The X coordinate of the center of the bounding box.
    y_center: float
        The Y coordinate of the center of the bounding box.
    width: float
        The width of the bounding box.
    height: float
        The height of the bounding box.
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

    @property
    def y_center(self) -> float:
        """Returns the Y coordinate of the center of the bounding box.

        @return: Y coordinate of the center of the bounding box.
        @rtype: float
        """
        return self._y_center

    @property
    def width(self) -> float:
        """Returns the width of the bounding box.

        @return: Width of the bounding box.
        @rtype: float
        """
        return self._width

    @property
    def height(self) -> float:
        """Returns the height of the bounding box.

        @return: Height of the bounding box.
        @rtype: float
        """
        return self._height

    @property
    def angle(self) -> float:
        """Returns the angle of the bounding box.

        @return: Angle of the bounding box.
        @rtype: float
        """
        return self._angle

    @property
    def confidence(self) -> float:
        """Returns the confidence of the detection.

        @return: Confidence of the detection.
        @rtype: float
        """
        return self._confidence

    @property
    def label(self) -> int:
        """Returns the label of the detection.

        @return: Label of the detection.
        @rtype: int
        """
        return self._label

    @property
    def keypoints(
        self,
    ) -> List[Keypoint]:
        """Returns the keypoints.

        @return: List of keypoints.
        @rtype: Keypoints
        """
        return self._keypoints

    @x_center.setter
    def x_center(self, value: float):
        """Sets the X coordinate of the center of the bounding box.

        @param value: X coordinate of the center of the bounding box.
        @type value: float
        @raise TypeError: If the X coordinate is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("X center must be a float.")

        self._x_center = value

    @y_center.setter
    def y_center(self, value: float):
        """Sets the Y coordinate of the center of the bounding box.

        @param value: Y coordinate of the center of the bounding box.
        @type value: float
        @raise TypeError: If the Y coordinate is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("Y center must be a float.")

        self._y_center = value

    @width.setter
    def width(self, value: float):
        """Sets the width of the bounding box.

        @param value: Width of the bounding box.
        @type value: float
        @raise TypeError: If the width is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("Width must be a float.")

        self._width = value

    @height.setter
    def height(self, value: float):
        """Sets the height of the bounding box.

        @param value: Height of the bounding box.
        @type value: float
        @raise TypeError: If the height is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("Height must be a float.")

        self._height = value

    @angle.setter
    def angle(self, value: float):
        """Sets the angle of the bounding box.

        @param value: Angle of the bounding box.
        @type value: float
        @raise TypeError: If the angle is not between -360 and 360.
        """
        if not isinstance(value, float):
            raise TypeError("Angle must be a float.")

        if value < -360 or value > 360:
            raise TypeError("Angle must be between -360 and 360 degrees.")

        self._angle = value

    @confidence.setter
    def confidence(self, value: float):
        """Sets the confidence of the detection.

        @param value: Confidence of the detection.
        @type value: float
        """
        if not isinstance(value, float):
            raise TypeError("Confidence must be a float.")

        self._confidence = value

    @label.setter
    def label(self, value: int):
        """Sets the label of the detection.

        @param value: Label of the detection.
        @type value: int
        """
        if not isinstance(value, int):
            raise TypeError("Label must be an integer.")

        self._label = value

    @keypoints.setter
    def keypoints(
        self,
        keypoints: List[Keypoint],
    ) -> None:
        """Sets the keypoints.

        @param value: List of keypoints.
        @type value: List[Keypoint]
        """
        if not isinstance(keypoints, list) or not all(
            isinstance(kp, Keypoint) for kp in keypoints
        ):
            raise ValueError("Keypoints must be a list of Keypoint objects.")

        self._keypoints = keypoints


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
        masks: np.ndarray
            The segmentation masks of the image. All masks are stored in a single numpy array.
        """
        super().__init__()
        self._detections: List[ImgDetectionExtended] = []
        self._masks: np.ndarray = np.array([])

    @property
    def detections(self) -> List[ImgDetectionExtended]:
        """Returns the image detections with keypoints.

        @return: List of image detections with keypoints.
        @rtype: List[ImgDetectionExtended]
        """
        return self._detections

    @property
    def masks(self) -> np.ndarray:
        """Returns the masks.

        @return: Masks.
        @rtype: np.ndarray
        """
        return self._masks

    @detections.setter
    def detections(self, detections: List[ImgDetectionExtended]):
        """Sets the image detections with keypoints.

        @param value: List of image detections with keypoints.
        @type value: List[ImgDetectionExtended]
        @raise TypeError: If the detections are not a list.
        @raise TypeError: If each detection is not an instance of ImgDetectionExtended.
        """
        if not isinstance(detections, list):
            raise TypeError("Detections must be a list")
        if not all(isinstance(item, ImgDetectionExtended) for item in detections):
            raise TypeError(
                "Each detection must be an instance of ImgDetectionExtended"
            )
        self._detections = detections

    @masks.setter
    def masks(self, masks: np.ndarray):
        """Sets the masks of the image.

        @param masks: Mask coefficients.
        @type value: np.ndarray
        @raise TypeError: If the mask is not a numpy array.
        """
        if not isinstance(masks, np.ndarray):
            raise TypeError("Mask must be a numpy array")

        # if not np.all(masks >= 0 or masks <= 1):
        # raise ValueError(f"Masks should be in range [0, 1], got {masks}.")

        self._masks = masks
