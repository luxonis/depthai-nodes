from typing import List, Tuple

import depthai as dai
import numpy as np
from numpy.typing import NDArray

from depthai_nodes.ml.helpers.constants import (
    KEYPOINT_COLOR,
)
from depthai_nodes.ml.messages.keypoints import Keypoint
from depthai_nodes.ml.messages.segmentation import SegmentationMask
from depthai_nodes.utils import get_logger
from depthai_nodes.utils.annotation_helper import AnnotationHelper


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
        transformation = self.transformation

        w, h = transformation.getSize()
        ratio = w / h
        border_thickness = 0.00333 * (h + w)  # TODO: improve
        text_size = 0.014 * (w + h)  # TODO: improve
        highlight_len = 0.035

        debug_color = (0, 1, 0, 1)
        item_fill_color = (21 / 255, 127 / 255, 88 / 255, 0.2)
        outline_color = (21 / 255, 127 / 255, 88 / 255, 1)

        annotation_builder = AnnotationHelper()
        for detection in self.detections:
            # TODO: refactor and use constants
            x_min, y_min, x_max, y_max = tuple(detection.rotated_rect.getOuterRect())
            # highlight_size_x = min(highlight_len, x_max - x_min)
            highlight_size_y = min(highlight_len * ratio, y_max - y_min)

            # TODO: draw just the visible part
            annotation_builder.draw_rotated_rect(  # Draws the outline
                center=(
                    detection.rotated_rect.center.x,
                    detection.rotated_rect.center.y,
                ),
                size=(
                    detection.rotated_rect.size.width,
                    detection.rotated_rect.size.height,
                ),
                angle=detection.rotated_rect.angle,
                outline_color=debug_color,
                fill_color=item_fill_color,
                thickness=0,
                clip_to_viewport=True,
            )

            pts: List[dai.Point2f] = detection.rotated_rect.getPoints()
            pts_len = len(pts)
            for i in range(pts_len):
                previous_pt = pts[(i - 1) % pts_len]
                current_pt = pts[i]
                next_pt = pts[(i + 1) % pts_len]

                corner_to_previous_pt = self._get_partial_line(
                    start_point=(current_pt.x, current_pt.y),
                    direction_point=(previous_pt.x, previous_pt.y),
                    length=highlight_size_y,  # TODO: improve the calculation
                )
                corner_to_next_pt = self._get_partial_line(
                    start_point=(current_pt.x, current_pt.y),
                    direction_point=(next_pt.x, next_pt.y),
                    length=highlight_size_y,  # TODO: improve the calculation
                )
                annotation_builder.draw_line(
                    corner_to_previous_pt,
                    (current_pt.x, current_pt.y),
                    color=outline_color,
                    thickness=border_thickness,
                    clip_to_viewport=True,
                )
                annotation_builder.draw_line(
                    (current_pt.x, current_pt.y),
                    corner_to_next_pt,
                    color=outline_color,
                    thickness=border_thickness,
                    clip_to_viewport=True,
                )

            text_space = text_size / 2 / h  # TODO: abstract to an object
            text_position = min(pts, key=lambda pt: (pt.y, pt.x))
            text_position_y = text_position.y - text_space
            relative_text_size = text_size / h
            if text_position_y - relative_text_size < 0:
                text_position = min(pts, key=lambda pt: (-pt.y, pt.x))
                text_position_y = text_position.y + text_space + relative_text_size
            text_position_x = max(min(text_position.x, 1), 0)

            annotation_builder.draw_text(  # Draws label text
                text=f"{detection.label_name} {int(detection.confidence * 100)}%",
                position=(
                    text_position_x,
                    text_position_y,
                ),
                color=(1, 1, 1, 1),
                background_color=(0, 0, 0, 0),
                size=text_size,
            )

            if any(detection.keypoints):
                keypoints = [
                    (keypoint.x, keypoint.y) for keypoint in detection.keypoints
                ]
                annotation_builder.draw_points(
                    points=keypoints,
                    color=(
                        KEYPOINT_COLOR.r,
                        KEYPOINT_COLOR.g,
                        KEYPOINT_COLOR.b,
                        KEYPOINT_COLOR.a,
                    ),
                    thickness=2,
                )

        return annotation_builder.build(self.getTimestamp(), self.getSequenceNum())

    # TODO: move the method to AnnotationsBuilder?
    def _get_partial_line(
        self,
        start_point: Tuple[float, float],
        direction_point: Tuple[float, float],
        length: float,
    ) -> Tuple[float, float]:
        """Calculate endpoint for a line starting at start_point going towards
        direction_point with specified length.

        Args:
            start_point: Starting point (x, y)
            direction_point: Point that defines direction (x, y)
            length: Desired length of the line

        Returns:
            Endpoint coordinates (x, y)
        """
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
