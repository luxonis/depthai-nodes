import copy
from typing import List, Tuple, Union

import depthai as dai
import numpy as np
from numpy.typing import NDArray

from depthai_nodes import (
    BACKGROUND_COLOR,
    KEYPOINT_COLOR,
    OUTLINE_COLOR,
    TEXT_COLOR,
)
from depthai_nodes.logging import get_logger

from .keypoints import Keypoint, Keypoints
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
    keypoints: Keypoints
        Keypoints of the detection. Getter returns list of Keypoint objects and setter accepts Keypoints object.
    """

    def __init__(self):
        """Initializes the ImgDetectionExtended object."""
        super().__init__()
        self._rotated_rect: dai.RotatedRect
        self._confidence: float = -1.0
        self._label: int = -1
        self._label_name: str = ""
        self._keypoints: Keypoints = Keypoints()
        self._logger = get_logger(__name__)

    def copy(self):
        """Creates a new instance of the ImgDetectionExtended class and copies the
        attributes.

        @return: A new instance of the ImgDetectionExtended class.
        @rtype: ImgDetectionExtended
        """
        new_obj = ImgDetectionExtended()
        rectangle = (
            self._rotated_rect.center.x,
            self._rotated_rect.center.y,
            self._rotated_rect.size.width,
            self._rotated_rect.size.height,
            self._rotated_rect.angle,
        )
        new_obj.rotated_rect = copy.deepcopy(rectangle)
        new_obj.confidence = copy.deepcopy(self.confidence)
        new_obj.label = copy.deepcopy(self.label)
        new_obj.label_name = copy.deepcopy(self.label_name)
        new_kpts_msg = Keypoints()
        new_kpts = []
        for kpt in self.keypoints:
            new_kpt = Keypoint()
            new_kpt.x = kpt.x
            new_kpt.y = kpt.y
            new_kpt.z = kpt.z
            new_kpt.confidence = kpt.confidence
            new_kpt.label = kpt.label
            new_kpts.append(new_kpt)
        new_kpts_msg.keypoints = new_kpts
        new_kpts_msg.edges = copy.deepcopy(self._keypoints.edges)
        new_obj.keypoints = new_kpts_msg
        return new_obj

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
        center = dai.Point2f(rectangle[0], rectangle[1], normalized=True)
        size = dai.Size2f(rectangle[2], rectangle[3], normalized=True)

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
        if value < -0.1 or value > 1.1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not (0 <= value <= 1):
            value = max(0, min(1, value))
            self._logger.info("Confidence value was clipped to [0, 1].")

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
        return self._keypoints.keypoints

    @keypoints.setter
    def keypoints(
        self,
        value: Keypoints,
    ) -> None:
        """Sets the keypoints.

        @param value: Keypoints object.
        @type value: Keypoints
        @raise TypeError: If value is not a Keypoints object.
        """
        if not isinstance(value, Keypoints):
            raise TypeError("Keypoints must be a Keypoints object.")
        self._keypoints = value


class ImgDetectionsExtended(dai.Buffer):
    """ImgDetectionsExtended class for storing image detections with keypoints.

    Attributes
    ----------
    detections: List[ImgDetectionExtended]
        Image detections with keypoints.
    masks: np.ndarray
        The segmentation masks of the image. All masks are stored in a single numpy array.
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self) -> None:
        """Initializes the ImgDetectionsExtended object."""
        super().__init__()
        self._detections: List[ImgDetectionExtended] = []
        self._masks: SegmentationMask = SegmentationMask()
        self._transformation: dai.ImgTransformation = None

    def copy(self):
        """Creates a new instance of the ImgDetectionsExtended class and copies the
        attributes.

        @return: A new instance of the ImgDetectionsExtended class.
        @rtype: ImgDetectionsExtended
        """
        new_obj = ImgDetectionsExtended()
        new_obj.detections = [det.copy() for det in self.detections]
        new_obj.masks = self._masks.copy()
        new_obj.transformation = self.transformation
        new_obj.setSequenceNum(self.getSequenceNum())
        new_obj.setTimestamp(self.getTimestamp())
        new_obj.setTimestampDevice(self.getTimestampDevice())
        new_obj.setTransformation(self.transformation)
        return new_obj

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
    def masks(self, value: Union[NDArray[np.int16], SegmentationMask]):
        """Sets the segmentation mask.

        @param value: Segmentation mask.
        @type value: NDArray[np.int8])
        @raise TypeError: If value is not a numpy array.
        @raise ValueError: If value is not a 2D numpy array.
        @raise ValueError: If each element is not of type int8.
        @raise ValueError: If each element is larger or equal to -1.
        """
        if isinstance(value, SegmentationMask):
            self._masks = value
        elif isinstance(value, np.ndarray):
            if not (value.size == 0 or value.ndim == 2):
                raise ValueError("Mask must be 2D.")
            if value.dtype != np.int16:
                raise ValueError("Mask must be an array of int16.")
            if np.any((value < -1)):
                raise ValueError("Mask must be an array values larger or equal to -1.")
            masks_msg = SegmentationMask()
            masks_msg.mask = value
            self._masks = masks_msg
        else:
            raise TypeError("Mask must be a numpy array or a SegmentationMask object.")

    @property
    def transformation(self) -> dai.ImgTransformation:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self._transformation

    @transformation.setter
    def transformation(self, value: dai.ImgTransformation):
        """Sets the Image Transformation object.

        @param value: The Image Transformation object.
        @type value: dai.ImgTransformation
        @raise TypeError: If value is not a dai.ImgTransformation object.
        """
        if value is not None:
            if not isinstance(value, dai.ImgTransformation):
                raise TypeError(
                    f"Transformation must be a dai.ImgTransformation object, instead got {type(value)}."
                )
        self._transformation = value

    def setTransformation(self, transformation: dai.ImgTransformation):
        """Sets the Image Transformation object.

        @param transformation: The Image Transformation object.
        @type transformation: dai.ImgTransformation
        @raise TypeError: If value is not a dai.ImgTransformation object.
        """
        if transformation is not None:
            assert isinstance(transformation, dai.ImgTransformation)
        self.transformation = transformation

    def getTransformation(self) -> dai.ImgTransformation:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self.transformation

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        img_annotations = dai.ImgAnnotations()
        annotation = dai.ImgAnnotation()
        transformation = self.transformation
        w, h = transformation.getSize()

        for detection in self.detections:
            detection: ImgDetectionExtended = detection
            rotated_rect = detection.rotated_rect
            rotated_rect = rotated_rect.denormalize(w, h)
            points = rotated_rect.getPoints()
            points = [dai.Point2f(point.x / w, point.y / h) for point in points]
            pointsAnnotation = dai.PointsAnnotation()
            pointsAnnotation.type = dai.PointsAnnotationType.LINE_LOOP
            pointsAnnotation.points = dai.VectorPoint2f(points)
            pointsAnnotation.outlineColor = OUTLINE_COLOR
            pointsAnnotation.thickness = 2.0
            annotation.points.append(pointsAnnotation)

            text = dai.TextAnnotation()
            text.position = points[0]
            text.text = f"{detection.label_name} {int(detection.confidence * 100)}%"
            text.fontSize = 15
            text.textColor = TEXT_COLOR
            text.backgroundColor = BACKGROUND_COLOR
            annotation.texts.append(text)

            if len(detection.keypoints) > 0:
                keypoints = [
                    dai.Point2f(keypoint.x, keypoint.y)
                    for keypoint in detection.keypoints
                ]
                keypointAnnotation = dai.PointsAnnotation()
                keypointAnnotation.type = dai.PointsAnnotationType.POINTS
                keypointAnnotation.points = dai.VectorPoint2f(keypoints)
                keypointAnnotation.outlineColor = KEYPOINT_COLOR
                keypointAnnotation.fillColor = KEYPOINT_COLOR
                keypointAnnotation.thickness = 2
                annotation.points.append(keypointAnnotation)

                if detection._keypoints.edges is not None:
                    for edge in detection._keypoints.edges:
                        skeletonAnnotation = dai.PointsAnnotation()
                        skeletonAnnotation.type = dai.PointsAnnotationType.LINE_STRIP
                        pt1 = keypoints[edge[0]]
                        pt2 = keypoints[edge[1]]
                        skeletonAnnotation.points = dai.VectorPoint2f(
                            [dai.Point2f(pt1.x, pt1.y), dai.Point2f(pt2.x, pt2.y)]
                        )
                        skeletonAnnotation.outlineColor = KEYPOINT_COLOR
                        skeletonAnnotation.fillColor = KEYPOINT_COLOR
                        skeletonAnnotation.thickness = 1
                        annotation.points.append(skeletonAnnotation)

        img_annotations.annotations.append(annotation)
        img_annotations.setTimestamp(self.getTimestamp())
        return img_annotations
