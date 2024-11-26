from typing import List, Tuple

import depthai as dai
import numpy as np
from numpy.typing import NDArray

from depthai_nodes.ml.helpers.constants import (
    BACKGROUND_COLOR,
    KEYPOINT_COLOR,
    OUTLINE_COLOR,
    TEXT_COLOR,
)

from .keypoints import Keypoint
from .segmentation import SegmentationMask


class ImgDetectionExtended(dai.Buffer):
    """A class for storing image detections in (x_center, y_center, width, height)
    format with additional angle, label and keypoints.

    Attributes
    ----------
    rotated_rect: dai.RotatedRect
        Rotated rectangle object defined by the center, width, height and angle in degrees.
    confidence: float
        Confidence of the detection.
    label: int
        Label of the detection.
    label_name: str
        The corresponding label name if available.
    keypoints: List[Keypoint]
        Keypoints of the detection.
    """

    def __init__(self):
        """Initializes the ImgDetectionExtended object."""
        super().__init__()
        self._rotated_rect: dai.RotatedRect
        self._confidence: float = -1.0
        self._label: int = -1
        self._label_name: str = ""
        self._keypoints: List[Keypoint] = []

    @property
    def rotated_rect(self) -> dai.RotatedRect:
        """Returns the rotated rectangle representing the bounding box.

        @return: Rotated rectangle object
        @rtype: dai.RotatedRect
        """
        return self._rotated_rect

    @rotated_rect.setter
    def rotated_rect(self, rectangle: Tuple[float, float, float, float, float]):
        """Sets the rotated rectangle of the bounding box.

        @param value: Tuple of (x_center, y_center, width, height, angle).
        @type value: tuple[float, float, float, float, float]
        """
        center = dai.Point2f(rectangle[0], rectangle[1])
        size = dai.Size2f(rectangle[2], rectangle[3])

        self._rotated_rect = dai.RotatedRect(center, size, rectangle[4])

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
        if value < 0 or value > 1:
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
    def label_name(self) -> str:
        """Returns the label name of the detection.

        @return: Label name of the detection.
        @rtype: str
        """
        return self._label_name

    @label_name.setter
    def label_name(self, value: str):
        """Sets the label name of the detection.

        @param value: Label name of the detection.
        @type value: str
        @raise TypeError: If value is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("Label name must be a string.")
        self._label_name = value

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
        """Initializes the ImgDetectionsExtended object."""
        super().__init__()
        self._detections: List[ImgDetectionExtended] = []
        self._masks: SegmentationMask = SegmentationMask()

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
    def masks(self) -> NDArray[np.int16]:
        """Returns the segmentation masks stored in a single numpy array.

        @return: Segmentation masks.
        @rtype: SegmentationMask
        """
        return self._masks.mask

    @masks.setter
    def masks(self, value: NDArray[np.int16]):
        """Sets the segmentation mask.

        @param value: Segmentation mask.
        @type value: NDArray[np.int8])
        @raise TypeError: If value is not a numpy array.
        @raise ValueError: If value is not a 2D numpy array.
        @raise ValueError: If each element is not of type int8.
        @raise ValueError: If each element is larger or equal to -1.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Mask must be a numpy array.")
        if value.ndim != 2:
            raise ValueError("Mask must be 2D.")
        if value.dtype != np.int16:
            raise ValueError("Mask must be an array of int16.")
        if np.any((value < -1)):
            raise ValueError("Mask must be an array values larger or equal to -1.")
        masks_msg = SegmentationMask()
        masks_msg.mask = value
        self._masks = masks_msg

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        """Returns default visualization message for the detections.

        The message adds the bounding boxes, labels and keypoints to the image
        annotations.
        """
        img_annotations = dai.ImgAnnotations()
        annotation = dai.ImgAnnotation()

        for detection in self.detections:
            detection: ImgDetectionExtended = detection
            rotated_rect = detection.rotated_rect
            points = rotated_rect.getPoints()
            pointsAnnotation = dai.PointsAnnotation()
            pointsAnnotation.type = dai.PointsAnnotationType.LINE_STRIP
            pointsAnnotation.points = dai.VectorPoint2f(points)
            pointsAnnotation.outlineColor = OUTLINE_COLOR
            pointsAnnotation.thickness = 2.0
            annotation.points.append(pointsAnnotation)

            text = dai.TextAnnotation()
            text.position = points[0]
            text.text = f"{detection.label_name} {int(detection.confidence * 100)}%"
            text.fontSize = 50.5
            text.textColor = TEXT_COLOR
            text.backgroundColor = BACKGROUND_COLOR
            annotation.texts.append(text)

            if len(detection.keypoints) > 0:
                keypoints = [
                    dai.Point2f(keypoint.x, keypoint.y)
                    for keypoint in detection.keypoints
                ]
                keypointAnnotation = dai.PointsAnnotation()
                keypointAnnotation.type = (
                    dai.PointsAnnotationType.LINE_STRIP
                )  # change to POINTS when fixed
                keypointAnnotation.points = dai.VectorPoint2f(keypoints)
                keypointAnnotation.outlineColor = KEYPOINT_COLOR
                keypointAnnotation.thickness = 2.0
                annotation.points.append(keypointAnnotation)

        img_annotations.annotations.append(annotation)
        img_annotations.setTimestamp(self.getTimestamp())
        return img_annotations
