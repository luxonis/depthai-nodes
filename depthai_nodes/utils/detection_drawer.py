import math
from typing import TYPE_CHECKING, List, Tuple

import depthai as dai
import numpy as np

from depthai_nodes import (
    FONT_BACKGROUND_COLOR,
    FONT_COLOR,
    KEYPOINT_COLOR,
    PRIMARY_COLOR,
    SMALLER_DETECTION_THRESHOLD,
    TRANSPARENT_PRIMARY_COLOR,
)
from depthai_nodes.utils.annotation_helper import AnnotationHelper
from depthai_nodes.utils.annotation_sizes import AnnotationSizes
from depthai_nodes.utils.smaller_annotation_sizes import SmallerAnnotationSizes

if TYPE_CHECKING:
    from depthai_nodes import ImgDetectionExtended


class DetectionDrawer:
    """A utility class for drawing visual annotations of object detections on images.

    This class provides functionality to render complete visual representations of object
    detections including bounding boxes, labels, confidence scores, and keypoints. It handles
    both regular and rotated bounding boxes, automatically adjusting annotation sizes based
    on detection dimensions.

    The drawer supports the following visual components:
    - Semi-transparent filled overlays for bounding boxes
    - Corner lines at each corner of rotated rectangles
    - Label text with confidence percentages
    - Keypoints with optional connecting edges
    - Adaptive sizing for small detections

    Attributes
    ----------
    DETECTION_CORNER_COLOR : dai.Color
        Color used for drawing corner lines of bounding boxes.
    DETECTION_FILL_COLOR : dai.Color
        Semi-transparent color used for filling bounding box overlays.
    """

    DETECTION_CORNER_COLOR = PRIMARY_COLOR
    DETECTION_FILL_COLOR = TRANSPARENT_PRIMARY_COLOR

    def __init__(
        self, annotation_helper: AnnotationHelper, size: Tuple[int, int]
    ) -> None:
        """Initializes the DetectionDrawer with an annotation helper and image
        dimensions.

        @param annotation_helper: Helper object for drawing annotations on the image
        @type annotation_helper: AnnotationHelper
        @param size: Image dimensions as (width, height) tuple
        @type size: Tuple[int, int]
        """
        self._annotation_helper = annotation_helper
        self._width, self._height = size

    def draw(self, detection: "ImgDetectionExtended") -> None:
        """Draws a detection on the image with all its visual components.

        This method renders the complete visual representation of a detection including:
            - Semi-transparent filled overlay of the bounding box
            - Corner lines at each corner of the rotated rectangle
            - Label text with confidence percentage
            - Keypoints and their connecting edges (if available)

        The annotation size is automatically adjusted based on the detection size,
        using smaller annotations for detections below the threshold defined by
        SMALLER_DETECTION_THRESHOLD.

        @param detection: The detection to draw with bounding box, confidence, label, and optional keypoints
        @type detection: ImgDetectionExtended
        """
        annotation_sizes = (
            AnnotationSizes(self._width, self._height)
            if not self._is_small_detection(detection)
            else SmallerAnnotationSizes(self._width, self._height)
        )

        self._draw_overlay(detection)
        self._draw_corners(annotation_sizes, detection)
        self._draw_label(annotation_sizes, detection)
        if any(detection.keypoints):
            self._draw_keypoints(annotation_sizes, detection)

    def _is_small_detection(self, detection: "ImgDetectionExtended") -> bool:
        size = detection.rotated_rect.size
        return (
            size.width <= SMALLER_DETECTION_THRESHOLD
            or size.height <= SMALLER_DETECTION_THRESHOLD
        )

    def _draw_overlay(self, detection: "ImgDetectionExtended") -> None:
        self._annotation_helper.draw_rotated_rect(
            center=detection.rotated_rect.center,
            size=detection.rotated_rect.size,
            angle=detection.rotated_rect.angle,
            fill_color=self.DETECTION_FILL_COLOR,
            thickness=0,
            clip_to_viewport=True,
        )

    def _draw_corners(
        self, annotation_sizes: AnnotationSizes, detection: "ImgDetectionExtended"
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
                )
                / 2,
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
                )
                / 2,
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
            self._annotation_helper.draw_line(
                corner_to_previous_pt,
                current_pt,
                color=self.DETECTION_CORNER_COLOR,
                thickness=annotation_sizes.border_thickness,
                clip_to_viewport=True,
            )
            self._annotation_helper.draw_line(
                current_pt,
                corner_to_next_pt,
                color=self.DETECTION_CORNER_COLOR,
                thickness=annotation_sizes.border_thickness,
                clip_to_viewport=True,
            )

    def _draw_label(
        self, annotation_sizes: AnnotationSizes, detection: "ImgDetectionExtended"
    ) -> None:
        pts = self._get_points(detection)
        text_position = min(pts, key=lambda pt: (pt.y, pt.x))
        if detection.rotated_rect.angle == 0:
            text_position_x = text_position.x + annotation_sizes.font_space
            text_position_y = (
                text_position.y
                + annotation_sizes.font_space
                + annotation_sizes.relative_font_size
            )
        else:
            text_position_x = text_position.x
            text_position_y = text_position.y - annotation_sizes.font_space
            if text_position_y - annotation_sizes.relative_font_size < 0:
                text_position = min(pts, key=lambda pt: (-pt.y, pt.x))
                text_position_y = (
                    text_position.y
                    + annotation_sizes.font_space
                    + annotation_sizes.relative_font_size
                )
        text_position_x = max(min(text_position_x, 1), 0)

        self._annotation_helper.draw_text(
            text=f"{detection.label_name} {int(detection.confidence * 100)}%",
            position=(
                text_position_x,
                text_position_y,
            ),
            color=FONT_COLOR,
            background_color=FONT_BACKGROUND_COLOR,
            size=annotation_sizes.font_size,
        )

    def _draw_keypoints(
        self, annotation_sizes: AnnotationSizes, detection: "ImgDetectionExtended"
    ) -> None:
        keypoints = [(keypoint.x, keypoint.y) for keypoint in detection.keypoints]
        self._annotation_helper.draw_points(
            points=keypoints,
            color=KEYPOINT_COLOR,
            thickness=annotation_sizes.keypoint_thickness,
        )
        if detection.edges:
            self._draw_edges(detection)

    def _draw_edges(self, detection: "ImgDetectionExtended") -> None:
        for edge in detection.edges:
            pt1_ix, pt2_ix = edge
            pt1 = detection.keypoints[pt1_ix]
            pt2 = detection.keypoints[pt2_ix]
            self._annotation_helper.draw_line(
                pt1=(pt1.x, pt1.y),
                pt2=(pt2.x, pt2.y),
                color=PRIMARY_COLOR,
                thickness=1,
            )

    def _get_points(self, detection: "ImgDetectionExtended") -> List[dai.Point2f]:
        return detection.rotated_rect.getPoints()

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
        total_length = math.hypot(dx, dy)  # equivalent to sqrt(dx^2 + dy^2)

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
        return math.hypot(
            x2 - x1, y2 - y1
        )  # equivalent to sqrt((x2 - x1)^2 + (y2 - y1)^2)

    def _scale_to_aspect(
        self, pt1: Tuple[float, float], pt2: Tuple[float, float], aspect_ratio: float
    ) -> float:
        angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        return 1.0 + (aspect_ratio - 1.0) * abs(np.sin(angle))
