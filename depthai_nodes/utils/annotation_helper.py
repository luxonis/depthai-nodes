from datetime import timedelta
from typing import List, Optional, Tuple, Union

import depthai as dai
import numpy as np

from depthai_nodes.constants import (
    PRIMARY_COLOR,
    TRANSPARENT_PRIMARY_COLOR,
)
from depthai_nodes.utils.viewport_clipper import ViewportClipper

Point = Tuple[float, float]
ColorRGBA = Tuple[float, float, float, float]


class AnnotationHelper:
    """Simplifies `dai.ImgAnnotation` creation.

    After calling the desired drawing methods, call the `build` method to create the `ImgAnnotations` message.
    """

    def __init__(self, viewport_clipper: Optional[ViewportClipper] = None):
        self.annotation: dai.ImgAnnotation = dai.ImgAnnotation()
        if not viewport_clipper:
            viewport_clipper = ViewportClipper(
                min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0
            )
        self._viewport_clipper = viewport_clipper

    def draw_line(
        self,
        pt1: Union[Point, dai.Point2f],
        pt2: Union[Point, dai.Point2f],
        color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        thickness: float = 2.0,
        clip_to_viewport: bool = False,
    ) -> "AnnotationHelper":
        """Draws a line between two points.

        @param pt1: Start of the line
        @type pt1: Point | dai.Point2f
        @param pt2: End of the line
        @type pt2: Point | dai.Point2f
        @param color: Line color
        @type color: ColorRGBA | dai.Color
        @param thickness: Line thickness
        @type thickness: float
        @param clip_to_viewport: Indication whether to clip the line to the viewport
        @type clip_to_viewport: bool
        @return: self
        @rtype: AnnotationHelper
        """
        if not isinstance(pt1, dai.Point2f):
            pt1 = self._create_point(pt1)
        if not isinstance(pt2, dai.Point2f):
            pt2 = self._create_point(pt2)
        if clip_to_viewport:
            clipped = self._viewport_clipper.clip_line((pt1.x, pt1.y), (pt2.x, pt2.y))
            if not clipped:
                return self
            pt1 = self._create_point(clipped[0])
            pt2 = self._create_point(clipped[1])
        line = dai.PointsAnnotation()
        if not isinstance(color, dai.Color):
            color = self._create_color(color)
        line.fillColor = color
        line.outlineColor = color
        line.thickness = thickness
        line.type = dai.PointsAnnotationType.LINE_STRIP
        line.points = self._create_points_vector([pt1, pt2])
        self.annotation.points.append(line)
        return self

    def draw_polyline(
        self,
        points: Union[List[Point], List[dai.Point2f]],
        outline_color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        fill_color: Union[
            Optional[ColorRGBA], Optional[dai.Color]
        ] = TRANSPARENT_PRIMARY_COLOR,
        thickness: float = 1.0,
        closed: bool = False,
    ) -> "AnnotationHelper":
        """Draws a polyline.

        @param points: List of points of the polyline
        @type points: List[Point] | List[dai.Point2f]
        @param outline_color: Outline color
        @type outline_color: ColorRGBA | dai.Color
        @param fill_color: Fill color (None for no fill)
        @type fill_color: ColorRGBA | dai.Color | None
        @param thickness: Line thickness
        @type thickness: float
        @param closed: Creates polygon, instead of polyline if True
        @type closed: bool
        @return: self
        @rtype: AnnotationHelper
        """
        points_type = (
            dai.PointsAnnotationType.LINE_STRIP
            if not closed
            else dai.PointsAnnotationType.LINE_LOOP
        )
        if not all(isinstance(pt, dai.Point2f) for pt in points):
            points = [self._create_point(pt) for pt in points]
        if not isinstance(outline_color, dai.Color):
            outline_color = self._create_color(outline_color)
        if fill_color and not isinstance(fill_color, dai.Color):
            fill_color = self._create_color(fill_color)

        points_annot = self._create_points_annotation(
            points, outline_color, fill_color, points_type
        )
        points_annot.thickness = thickness
        self.annotation.points.append(points_annot)
        return self

    def draw_points(
        self,
        points: Union[List[Point], List[dai.Point2f], dai.VectorPoint2f],
        color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        thickness: float = 2.0,
    ) -> "AnnotationHelper":
        """Draws points.

        @param points: List of points to draw
        @type points: List[Point] | List[dai.Point2f]
        @param color: Color of the points
        @type color: ColorRGBA | dai.Color
        @param thickness: Size of the points
        @type thickness: float
        @return: self
        @rtype: AnnotationHelper
        """
        # TODO: Visualizer currently does not show dai.PointsAnnotationType.POINTS
        if not all(isinstance(pt, dai.Point2f) for pt in points):
            points = [self._create_point(pt) for pt in points]
        if not isinstance(color, dai.Color):
            color = self._create_color(color)
        points_annot = self._create_points_annotation(
            points, color, None, dai.PointsAnnotationType.POINTS
        )
        points_annot.thickness = thickness
        self.annotation.points.append(points_annot)
        return self

    def draw_circle(
        self,
        center: Union[Point, dai.Point2f],
        radius: float,
        outline_color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        fill_color: Union[Optional[ColorRGBA], Optional[dai.Color]] = None,
        thickness: float = 1.0,
    ) -> "AnnotationHelper":
        """Draws a circle.

        @param center: Center of the circle
        @type center: Point | dai.Point2f
        @param radius: Radius of the circle
        @type radius: float
        @param outline_color: Outline color
        @type outline_color: ColorRGBA | dai.Color
        @param fill_color: Fill color (None for no fill)
        @type fill_color: ColorRGBA | dai.Color | None
        @param thickness: Outline thickness
        @type thickness: float
        @return: self
        @rtype: AnnotationHelper
        """
        circle = dai.CircleAnnotation()
        if not isinstance(outline_color, dai.Color):
            outline_color = self._create_color(outline_color)
        circle.outlineColor = outline_color
        if fill_color is not None:
            if not isinstance(fill_color, dai.Color):
                fill_color = self._create_color(fill_color)
            circle.fillColor = fill_color
        circle.thickness = thickness
        circle.diameter = radius * 2
        if not isinstance(center, dai.Point2f):
            center = self._create_point(center)
        circle.position = center
        self.annotation.circles.append(circle)
        return self

    def draw_rectangle(
        self,
        top_left: Union[Point, dai.Point2f],
        bottom_right: Union[Point, dai.Point2f],
        outline_color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        fill_color: Union[
            Optional[ColorRGBA], Optional[dai.Color]
        ] = TRANSPARENT_PRIMARY_COLOR,
        thickness: float = 1.0,
        clip_to_viewport: bool = False,
    ) -> "AnnotationHelper":
        """Draws a rectangle.

        @param top_left: Top left corner of the rectangle
        @type top_left: Point | dai.Point2f
        @param bottom_right: Bottom right corner of the rectangle
        @type bottom_right: Point | dai.Point2f
        @param outline_color: Outline color
        @type outline_color: ColorRGBA | dai.Color
        @param fill_color: Fill color (None for no fill)
        @type fill_color: ColorRGBA | dai.Color | None
        @param thickness: Outline thickness
        @type thickness: float
        @param clip_to_viewport: Indication whether to clip the line to the viewport
        @type clip_to_viewport: bool
        @return: self
        @rtype: AnnotationHelper
        """
        if isinstance(top_left, dai.Point2f):
            top_left = (top_left.x, top_left.y)
        if isinstance(bottom_right, dai.Point2f):
            bottom_right = (bottom_right.x, bottom_right.y)

        points = [
            top_left,
            (bottom_right[0], top_left[1]),
            bottom_right,
            (top_left[0], bottom_right[1]),
        ]
        if clip_to_viewport:
            points = self._viewport_clipper.clip_rect(points)
        self.draw_polyline(points, outline_color, fill_color, thickness, closed=True)
        return self

    def draw_text(
        self,
        text: str,
        position: Union[Point, dai.Point2f],
        color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        background_color: Union[Optional[ColorRGBA], Optional[dai.Color]] = None,
        size: float = 32,
    ) -> "AnnotationHelper":
        """Draws text.

        @param text: Text string
        @type text: str
        @param position: Text position
        @type position: Point | dai.Point2f
        @param color: Text color
        @type color: ColorRGBA | dai.Color
        @param background_color: Background color (None for no background)
        @type background_color: ColorRGBA | dai.Color | None
        @param size: Text size
        @type size: float
        @return: self
        @rtype: AnnotationHelper
        """
        text_annot = dai.TextAnnotation()
        if not isinstance(position, dai.Point2f):
            position = self._create_point(position)
        text_annot.position = position
        text_annot.text = text
        if not isinstance(color, dai.Color):
            color = self._create_color(color)
        text_annot.textColor = color
        text_annot.fontSize = size
        if background_color is not None:
            if not isinstance(background_color, dai.Color):
                background_color = self._create_color(background_color)
            text_annot.backgroundColor = background_color
        self.annotation.texts.append(text_annot)
        return self

    def draw_rotated_rect(
        self,
        center: Union[Point, dai.Point2f],
        size: Union[Tuple[float, float], dai.Size2f],
        angle: float,
        outline_color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        fill_color: Union[
            Optional[ColorRGBA], Optional[dai.Color]
        ] = TRANSPARENT_PRIMARY_COLOR,
        thickness: float = 1.0,
        clip_to_viewport: bool = False,
    ) -> "AnnotationHelper":
        """Draws a rotated rectangle.

        @param center: Center of the rectangle
        @type center: Point | dai.Point2f
        @param size: Size of the rectangle (width, height)
        @type size: Tuple[float, float] | dai.Size2f
        @param angle: Angle of rotation in degrees
        @type angle: float
        @param outline_color: Outline color
        @type outline_color: ColorRGBA | dai.Color
        @param fill_color: Fill color (None for no fill)
        @type fill_color: ColorRGBA | dai.Color | None
        @param thickness: Outline thickness
        @type thickness: float
        @param clip_to_viewport: Indication whether to clip the line to the viewport
        @type clip_to_viewport: bool
        @return: self
        @rtype: AnnotationHelper
        """
        if not isinstance(center, dai.Point2f):
            center = self._create_point(center)
        if not isinstance(size, dai.Size2f):
            size = self._create_size(size)
        points = self._get_rotated_rect_points(center, size, angle)
        if clip_to_viewport:
            points = self._viewport_clipper.clip_rect(points)
        self.draw_polyline(points, outline_color, fill_color, thickness, True)
        return self

    def build(self, timestamp: timedelta, sequence_num: int) -> dai.ImgAnnotations:
        """Creates an ImgAnnotations message.

        @param timestamp: Message timestamp
        @type timestamp: timedelta
        @param sequence_num: Message sequence number
        @type sequence_num: int
        @return: Created ImgAnnotations message
        @rtype: dai.ImgAnnotations
        """
        annotations_msg = dai.ImgAnnotations()
        annotations_msg.annotations = dai.VectorImgAnnotation([self.annotation])
        annotations_msg.setTimestamp(timestamp)
        annotations_msg.setSequenceNum(sequence_num)
        return annotations_msg

    def _create_point(self, point: Point) -> dai.Point2f:
        return dai.Point2f(point[0], point[1], True)

    def _create_points_annotation(
        self,
        points: List[dai.Point2f],
        color: dai.Color,
        fill_color: Optional[dai.Color],
        type: dai.PointsAnnotationType,
    ) -> dai.PointsAnnotation:
        points_annot = dai.PointsAnnotation()
        points_annot.outlineColor = color
        if fill_color is not None:
            points_annot.fillColor = fill_color
        points_annot.type = type
        points_annot.points = self._create_points_vector(points)
        return points_annot

    def _create_color(self, color: ColorRGBA) -> dai.Color:
        c = dai.Color()
        c.a = color[3]
        c.r = color[0]
        c.g = color[1]
        c.b = color[2]
        return c

    def _get_rotated_rect_points(
        self, center: dai.Point2f, size: dai.Size2f, angle: float
    ) -> List[Point]:
        angle_rad = np.radians(angle)

        # Half-dimensions
        dx = size.width / 2
        dy = size.height / 2

        # Define the corners relative to the center
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])

        # Rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        # Rotate and translate the corners
        rotated_corners = corners @ rotation_matrix.T
        translated_corners = rotated_corners + np.array([center.x, center.y])

        # Convert to list of tuples
        return [tuple(corner) for corner in translated_corners.tolist()]

    def _create_points_vector(self, points: List[dai.Point2f]) -> dai.VectorPoint2f:
        return dai.VectorPoint2f(points)

    def _create_size(self, size: Tuple[float, float]):
        return dai.Size2f(size[0], size[1])
