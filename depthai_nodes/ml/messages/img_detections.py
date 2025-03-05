from typing import List, Tuple

import depthai as dai
import numpy as np
from numpy.typing import NDArray

from depthai_nodes.ml.messages.keypoints import Keypoint
from depthai_nodes.ml.messages.segmentation import SegmentationMask
from depthai_nodes.utils import get_logger
from depthai_nodes.utils.annotation_helper import AnnotationHelper
from depthai_nodes.utils.annotation_sizes import AnnotationSizes
from depthai_nodes.utils.constants import (
    DETECTION_CORNER_COLOR,
    DETECTION_FILL_COLOR,
    KEYPOINT_COLOR,
    OUTLINE_COLOR,
    TEXT_BACKGROUND_COLOR,
    TEXT_COLOR,
)
from depthai_nodes.utils.viewport_clipper import ViewportClipper


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
        self._logger = get_logger(__name__)

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
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self) -> None:
        """Initializes the ImgDetectionsExtended object."""
        super().__init__()
        self._detections: List[ImgDetectionExtended] = []
        self._masks: SegmentationMask = SegmentationMask()
        self._transformation: dai.ImgTransformation = None
        self._viewport_clipper = ViewportClipper(
            min_x=0.0, min_y=0.0, max_x=1.0, max_y=1.0
        )

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

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        w, h = self.transformation.getSize()
        annotation_sizes = AnnotationSizes(w, h)
        annotation_builder = AnnotationHelper(self._viewport_clipper)
        for detection in self.detections:
            self._draw_overlay(annotation_builder, detection)
            self._draw_corners(annotation_sizes, annotation_builder, detection)
            self._draw_label(annotation_sizes, annotation_builder, detection)
            if any(detection.keypoints):
                self._draw_keypoints(annotation_sizes, annotation_builder, detection)

        return annotation_builder.build(self.getTimestamp(), self.getSequenceNum())

    def _draw_overlay(
        self, annotation_builder: AnnotationHelper, detection: ImgDetectionExtended
    ) -> None:
        annotation_builder.draw_rotated_rect(
            center=detection.rotated_rect.center,
            size=detection.rotated_rect.size,
            angle=detection.rotated_rect.angle,
            outline_color=OUTLINE_COLOR,
            fill_color=DETECTION_FILL_COLOR,
            thickness=0,
            clip_to_viewport=True,
        )

    def _draw_corners(
        self,
        annotation_sizes: AnnotationSizes,
        annotation_builder: AnnotationHelper,
        detection: ImgDetectionExtended,
    ) -> None:
        pts = self._get_points(detection)
        pts_len = len(pts)
        for i in range(pts_len):
            previous_pt = pts[(i - 1) % pts_len]
            current_pt = pts[i]
            next_pt = pts[(i + 1) % pts_len]

            corner_size_to_previous = min(
                annotation_sizes.corner_size,
                self._calculate_distance(
                    (current_pt.x, current_pt.y), (previous_pt.x, previous_pt.y)
                ),
            )
            corner_size_to_previous *= self._scale_to_aspect(
                (current_pt.x, current_pt.y),
                (previous_pt.x, previous_pt.y),
                annotation_sizes.aspect_ratio,
            )
            corner_to_previous_pt = self._get_partial_line(
                start_point=(current_pt.x, current_pt.y),
                direction_point=(previous_pt.x, previous_pt.y),
                length=corner_size_to_previous,
            )

            corner_size_to_next = min(
                annotation_sizes.corner_size,
                self._calculate_distance(
                    (current_pt.x, current_pt.y), (next_pt.x, next_pt.y)
                ),
            )
            corner_size_to_next *= self._scale_to_aspect(
                (current_pt.x, current_pt.y),
                (next_pt.x, next_pt.y),
                annotation_sizes.aspect_ratio,
            )
            corner_to_next_pt = self._get_partial_line(
                start_point=(current_pt.x, current_pt.y),
                direction_point=(next_pt.x, next_pt.y),
                length=corner_size_to_next,
            )
            annotation_builder.draw_line(
                corner_to_previous_pt,
                current_pt,
                color=DETECTION_CORNER_COLOR,
                thickness=annotation_sizes.border_thickness,
                clip_to_viewport=True,
            )
            annotation_builder.draw_line(
                current_pt,
                corner_to_next_pt,
                color=DETECTION_CORNER_COLOR,
                thickness=annotation_sizes.border_thickness,
                clip_to_viewport=True,
            )

    def _draw_label(
        self,
        annotation_sizes: AnnotationSizes,
        annotation_builder: AnnotationHelper,
        detection: ImgDetectionExtended,
    ) -> None:
        pts = self._get_points(detection)
        text_position = min(pts, key=lambda pt: (pt.y, pt.x))
        if detection.rotated_rect.angle == 0:
            text_position_x = text_position.x + annotation_sizes.text_space
            text_position_y = (
                text_position.y
                + annotation_sizes.text_space
                + annotation_sizes.relative_text_size
            )
        else:
            text_position_x = text_position.x
            text_position_y = text_position.y - annotation_sizes.text_space
            if text_position_y - annotation_sizes.relative_text_size < 0:
                text_position = min(pts, key=lambda pt: (-pt.y, pt.x))
                text_position_y = (
                    text_position.y
                    + annotation_sizes.text_space
                    + annotation_sizes.relative_text_size
                )
        text_position_x = max(min(text_position_x, 1), 0)

        annotation_builder.draw_text(
            text=f"{detection.label_name} {int(detection.confidence * 100)}%",
            position=(
                text_position_x,
                text_position_y,
            ),
            color=TEXT_COLOR,
            background_color=TEXT_BACKGROUND_COLOR,
            size=annotation_sizes.text_size,
        )

    def _get_points(self, detection: ImgDetectionExtended) -> List[dai.Point2f]:
        return detection.rotated_rect.getPoints()

    def _draw_keypoints(
        self,
        annotation_sizes: AnnotationSizes,
        annotation_builder: AnnotationHelper,
        detection: ImgDetectionExtended,
    ) -> None:
        keypoints = [(keypoint.x, keypoint.y) for keypoint in detection.keypoints]
        annotation_builder.draw_points(
            points=keypoints,
            color=KEYPOINT_COLOR,
            thickness=annotation_sizes.keypoint_thickness,
        )

    # TODO: move the method to AnnotationsBuilder?
    def _get_partial_line(
        self,
        start_point: Tuple[float, float],
        direction_point: Tuple[float, float],
        length: float,
    ) -> Tuple[float, float]:
        # Calculate direction vector
        dx = direction_point[0] - start_point[0]
        dy = direction_point[1] - start_point[1]

        # Calculate the total length of the full line
        total_length = (dx**2 + dy**2) ** 0.5

        # If total_length is 0, return start_point to avoid division by zero
        if total_length == 0:
            return start_point

        # Calculate scaling factor
        scale = length / total_length

        # Calculate the endpoint
        end_x = start_point[0] + dx * scale
        end_y = start_point[1] + dy * scale

        return (end_x, end_y)

    def _calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        x1, y1 = point1
        x2, y2 = point2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _scale_to_aspect(
        self, pt1: Tuple[float, float], pt2: Tuple[float, float], aspect_ratio: float
    ) -> float:
        angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        return 1.0 + (aspect_ratio - 1.0) * abs(np.sin(angle))
