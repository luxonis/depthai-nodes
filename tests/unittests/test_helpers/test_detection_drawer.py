import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import depthai as dai

from depthai_nodes import (
    KEYPOINT_COLOR,
    PRIMARY_COLOR,
    TRANSPARENT_PRIMARY_COLOR,
)
from depthai_nodes.message.img_detections import ImgDetectionExtended
from depthai_nodes.message.keypoints import Keypoint, Keypoints
from depthai_nodes.utils.annotation_helper import AnnotationHelper
from depthai_nodes.utils.detection_drawer import DetectionDrawer

Point = Tuple[float, float]
ColorRGBA = Tuple[float, float, float, float]


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-15) -> bool:
    """Check if two floating-point numbers are approximately equal."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def points_equal(point1: Point, point2: Point) -> bool:
    """Check if two points are approximately equal."""
    return is_close(point1[0], point2[0]) and is_close(point1[1], point2[1])


def colors_equal(
    color1: Union[ColorRGBA, dai.Color, None], color2: Union[ColorRGBA, dai.Color, None]
) -> bool:
    """Check if two colors are approximately equal."""
    if color1 is None and color2 is None:
        return True
    if color1 is None or color2 is None:
        return False

    c1_tuple = color_to_tuple(color1) if not isinstance(color1, tuple) else color1
    c2_tuple = color_to_tuple(color2) if not isinstance(color2, tuple) else color2

    return all(is_close(a, b) for a, b in zip(c1_tuple, c2_tuple))


def color_to_tuple(
    color: Union[ColorRGBA, dai.Color],
) -> Tuple[float, float, float, float]:
    if isinstance(color, dai.Color):
        return (color.r, color.g, color.b, color.a)
    return color


@dataclass
class RotatedRectData:
    center: Union[Point, dai.Point2f]
    size: Union[Tuple[float, float], dai.Size2f]
    angle: float
    outline_color: Union[ColorRGBA, dai.Color]
    fill_color: Union[Optional[ColorRGBA], Optional[dai.Color]]
    thickness: float
    clip_to_viewport: bool

    def __eq__(self, other):
        if not isinstance(other, RotatedRectData):
            return False
        return (
            self._points_equal(self.center, other.center)
            and self._sizes_equal(self.size, other.size)
            and is_close(self.angle, other.angle)
            and self._colors_equal(self.outline_color, other.outline_color)
            and self._colors_equal(self.fill_color, other.fill_color)
            and is_close(self.thickness, other.thickness)
            and self.clip_to_viewport == other.clip_to_viewport
        )

    def _points_equal(self, point1, point2):
        p1 = (point1.x, point1.y) if isinstance(point1, dai.Point2f) else point1
        p2 = (point2.x, point2.y) if isinstance(point2, dai.Point2f) else point2
        return points_equal(p1, p2)

    def _sizes_equal(self, size1, size2):
        s1 = (size1.width, size1.height) if isinstance(size1, dai.Size2f) else size1
        s2 = (size2.width, size2.height) if isinstance(size2, dai.Size2f) else size2
        return is_close(s1[0], s2[0]) and is_close(s1[1], s2[1])

    def _colors_equal(self, color1, color2):
        return colors_equal(color1, color2)


@dataclass
class LineData:
    pt1: Point
    pt2: Point
    color: Union[ColorRGBA, dai.Color]
    thickness: float
    clip_to_viewport: bool

    def __eq__(self, other):
        if not isinstance(other, LineData):
            return False
        return (
            points_equal(self.pt1, other.pt1)
            and points_equal(self.pt2, other.pt2)
            and colors_equal(self.color, other.color)
            and is_close(self.thickness, other.thickness)
            and self.clip_to_viewport == other.clip_to_viewport
        )


@dataclass
class TextData:
    text: str
    position: Point
    color: Union[ColorRGBA, dai.Color]
    background_color: Union[Optional[ColorRGBA], Optional[dai.Color]]
    size: float

    def __eq__(self, other):
        if not isinstance(other, TextData):
            return False
        return (
            self.text == other.text
            and points_equal(self.position, other.position)
            and colors_equal(self.color, other.color)
            and colors_equal(self.background_color, other.background_color)
            and is_close(self.size, other.size)
        )


@dataclass
class PointData:
    center: Point
    radius: float
    color: Union[ColorRGBA, dai.Color]
    thickness: float

    def __eq__(self, other):
        if not isinstance(other, PointData):
            return False
        return (
            points_equal(self.center, other.center)
            and is_close(self.radius, other.radius)
            and colors_equal(self.color, other.color)
            and is_close(self.thickness, other.thickness)
        )


class TestAnnotationHelper(AnnotationHelper):
    def __init__(self):
        super().__init__()
        self.rotated_rects = []
        self.lines = []
        self.texts = []
        self.points = []

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
    ) -> "TestAnnotationHelper":
        self.rotated_rects.append(
            RotatedRectData(
                center=center,
                size=size,
                angle=angle,
                outline_color=outline_color,
                fill_color=fill_color,
                thickness=thickness,
                clip_to_viewport=clip_to_viewport,
            )
        )
        return self

    def draw_line(
        self,
        pt1: Union[Point, dai.Point2f],
        pt2: Union[Point, dai.Point2f],
        color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        thickness: float = 2.0,
        clip_to_viewport: bool = False,
    ) -> "TestAnnotationHelper":
        self.lines.append(
            LineData(
                pt1=self._convert_point(pt1),
                pt2=self._convert_point(pt2),
                color=color_to_tuple(color),
                thickness=thickness,
                clip_to_viewport=clip_to_viewport,
            )
        )
        return self

    def draw_text(
        self,
        text: str,
        position: Union[Point, dai.Point2f],
        color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        background_color: Union[Optional[ColorRGBA], Optional[dai.Color]] = None,
        size: float = 32,
    ) -> "TestAnnotationHelper":
        position_tuple = self._convert_point(position)

        self.texts.append(
            TextData(
                text=text,
                position=position_tuple,
                color=color,
                background_color=background_color,
                size=size,
            )
        )
        return self

    def draw_points(
        self,
        points: Union[List[Point], List[dai.Point2f], dai.VectorPoint2f],
        color: Union[ColorRGBA, dai.Color] = PRIMARY_COLOR,
        thickness: float = 2.0,
    ) -> "TestAnnotationHelper":
        for point in points:
            self.points.append(
                PointData(
                    center=self._convert_point(point),
                    radius=thickness,
                    color=color,
                    thickness=thickness,
                )
            )
        return self

    def _convert_point(self, point: Union[Point, dai.Point2f]) -> Point:
        if isinstance(point, dai.Point2f):
            return (point.x, point.y)
        return point


def create_detection(
    x_center: float, y_center: float, width: float, height: float, angle=0.0
):
    detection = ImgDetectionExtended()
    detection.rotated_rect = (x_center, y_center, width, height, angle)
    detection.confidence = 0.85
    detection.label = 1
    detection.label_name = "person"
    return detection


def test_draw_basic_detection():
    test_annotation_helper = TestAnnotationHelper()
    drawer = DetectionDrawer(test_annotation_helper, (800, 600))

    detection = create_detection(0.5, 0.5, 0.2, 0.15)
    drawer.draw(detection)

    assert test_annotation_helper.rotated_rects == [
        RotatedRectData(
            center=dai.Point2f(0.5, 0.5),
            size=dai.Size2f(0.2, 0.15),
            angle=0.0,
            outline_color=PRIMARY_COLOR,
            fill_color=TRANSPARENT_PRIMARY_COLOR,
            thickness=0,
            clip_to_viewport=True,
        )
    ]
    assert test_annotation_helper.lines == [
        LineData(
            pt1=(0.4000000059604645, 0.4783333452542623),
            pt2=(0.4000000059604645, 0.42500001192092896),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.4000000059604645, 0.42500001192092896),
            pt2=(0.44000000596046446, 0.42500001192092896),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.5600000238418579, 0.42500001192092896),
            pt2=(0.6000000238418579, 0.42500001192092896),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6000000238418579, 0.42500001192092896),
            pt2=(0.6000000238418579, 0.4783333452542623),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6000000238418579, 0.5216666547457377),
            pt2=(0.6000000238418579, 0.574999988079071),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6000000238418579, 0.574999988079071),
            pt2=(0.5600000238418579, 0.574999988079071),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.44000000596046446, 0.574999988079071),
            pt2=(0.4000000059604645, 0.574999988079071),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.4000000059604645, 0.574999988079071),
            pt2=(0.4000000059604645, 0.5216666547457377),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
    ]


def test_draw_rotated_detection():
    test_annotation_helper = TestAnnotationHelper()
    drawer = DetectionDrawer(test_annotation_helper, (1920, 1080))

    detection = create_detection(0.4, 0.6, 0.3, 0.4, 45.0)
    drawer.draw(detection)

    assert test_annotation_helper.rotated_rects == [
        RotatedRectData(
            center=dai.Point2f(0.4, 0.6),
            size=dai.Size2f(0.3, 0.4),
            angle=45.0,
            outline_color=PRIMARY_COLOR,
            fill_color=TRANSPARENT_PRIMARY_COLOR,
            thickness=0,
            clip_to_viewport=True,
        )
    ]
    assert test_annotation_helper.lines == [
        LineData(
            pt1=(0.3915155362381454, 0.3963524797516416),
            pt2=(0.43535536527633667, 0.3525126576423645),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.43535536527633667, 0.3525126576423645),
            pt2=(0.4791951920793541, 0.39635248444538196),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6036475751592628, 0.5208048675252906),
            pt2=(0.6474874019622803, 0.5646446943283081),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6474874019622803, 0.5646446943283081),
            pt2=(0.6036475721790312, 0.6084845148730053),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.40848447642784136, 0.8036475694966542),
            pt2=(0.3646446466445923, 0.8474873900413513),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.3646446466445923, 0.8474873900413513),
            pt2=(0.3208048208349857, 0.803647561152227),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.1963524506694165, 0.679195182244532),
            pt2=(0.15251262485980988, 0.6353553533554077),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.15251262485980988, 0.6353553533554077),
            pt2=(0.19635245389800118, 0.5915155312461307),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
    ]


def test_draw_small_detection():
    test_annotation_helper = TestAnnotationHelper()
    drawer = DetectionDrawer(test_annotation_helper, (640, 480))

    detection = create_detection(0.125, 0.12, 0.05, 0.04)
    drawer.draw(detection)

    assert test_annotation_helper.rotated_rects == [
        RotatedRectData(
            center=dai.Point2f(0.125, 0.12),
            size=dai.Size2f(0.05, 0.04),
            angle=0.0,
            outline_color=PRIMARY_COLOR,
            fill_color=TRANSPARENT_PRIMARY_COLOR,
            thickness=0,
            clip_to_viewport=True,
        )
    ]
    assert test_annotation_helper.lines == [
        LineData(
            pt1=(0.10000000149011612, 0.1266666607062022),
            pt2=(0.10000000149011612, 0.09999999403953552),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=1.8666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.10000000149011612, 0.09999999403953552),
            pt2=(0.12000000149011612, 0.09999999403953552),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=1.8666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.1300000059604645, 0.09999999403953552),
            pt2=(0.15000000596046448, 0.09999999403953552),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=1.8666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.15000000596046448, 0.09999999403953552),
            pt2=(0.15000000596046448, 0.1266666607062022),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=1.8666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.15000000596046448, 0.11333333392937978),
            pt2=(0.15000000596046448, 0.14000000059604645),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=1.8666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.15000000596046448, 0.14000000059604645),
            pt2=(0.1300000059604645, 0.14000000059604645),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=1.8666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.12000000149011612, 0.14000000059604645),
            pt2=(0.10000000149011612, 0.14000000059604645),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=1.8666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.10000000149011612, 0.14000000059604645),
            pt2=(0.10000000149011612, 0.11333333392937978),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=1.8666666666666667,
            clip_to_viewport=True,
        ),
    ]


def test_draw_detection_with_keypoints():
    test_annotation_helper = TestAnnotationHelper()
    drawer = DetectionDrawer(test_annotation_helper, (1920, 1080))

    detection = create_detection(0.5, 0.5, 0.4, 0.6)

    keypoints_obj = Keypoints()
    keypoint1 = Keypoint()
    keypoint1.x = 0.4
    keypoint1.y = 0.3
    keypoint1.confidence = 0.9
    keypoint2 = Keypoint()
    keypoint2.x = 0.6
    keypoint2.y = 0.4
    keypoint2.confidence = 0.8
    keypoints_obj.keypoints = [keypoint1, keypoint2]

    detection.keypoints = keypoints_obj

    drawer.draw(detection)

    assert test_annotation_helper.rotated_rects == [
        RotatedRectData(
            center=dai.Point2f(0.5, 0.5),
            size=dai.Size2f(0.4, 0.6),
            angle=0.0,
            outline_color=PRIMARY_COLOR,
            fill_color=TRANSPARENT_PRIMARY_COLOR,
            thickness=0,
            clip_to_viewport=True,
        )
    ]
    assert test_annotation_helper.lines == [
        LineData(
            pt1=(0.30000001192092896, 0.27111109919018217),
            pt2=(0.30000001192092896, 0.19999998807907104),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.30000001192092896, 0.19999998807907104),
            pt2=(0.34000001192092894, 0.19999998807907104),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.659999988079071, 0.19999998807907104),
            pt2=(0.699999988079071, 0.19999998807907104),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.699999988079071, 0.19999998807907104),
            pt2=(0.699999988079071, 0.27111109919018217),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.699999988079071, 0.7288889008098178),
            pt2=(0.699999988079071, 0.800000011920929),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.699999988079071, 0.800000011920929),
            pt2=(0.659999988079071, 0.800000011920929),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.34000001192092894, 0.800000011920929),
            pt2=(0.30000001192092896, 0.800000011920929),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.30000001192092896, 0.800000011920929),
            pt2=(0.30000001192092896, 0.7288889008098178),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
    ]
    assert test_annotation_helper.points == [
        PointData(
            center=(0.4, 0.3),
            radius=10.0,
            color=KEYPOINT_COLOR,
            thickness=10.0,
        ),
        PointData(
            center=(0.6, 0.4),
            radius=10.0,
            color=KEYPOINT_COLOR,
            thickness=10.0,
        ),
    ]


def test_detection_with_empty_keypoints():
    test_annotation_helper = TestAnnotationHelper()
    drawer = DetectionDrawer(test_annotation_helper, (800, 600))

    detection = create_detection(0.35, 0.4, 0.3, 0.2)

    keypoints_obj = Keypoints()
    keypoints_obj.keypoints = []
    detection.keypoints = keypoints_obj

    drawer.draw(detection)

    assert test_annotation_helper.rotated_rects == [
        RotatedRectData(
            center=dai.Point2f(0.35, 0.4),
            size=dai.Size2f(0.3, 0.2),
            angle=0.0,
            outline_color=PRIMARY_COLOR,
            fill_color=TRANSPARENT_PRIMARY_COLOR,
            thickness=0,
            clip_to_viewport=True,
        )
    ]
    assert test_annotation_helper.lines == [
        LineData(
            pt1=(0.19999998807907104, 0.3533333452542623),
            pt2=(0.19999998807907104, 0.30000001192092896),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.19999998807907104, 0.30000001192092896),
            pt2=(0.23999998807907105, 0.30000001192092896),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.46, 0.30000001192092896),
            pt2=(0.5, 0.30000001192092896),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.5, 0.30000001192092896),
            pt2=(0.5, 0.3533333452542623),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.5, 0.44666666666666666),
            pt2=(0.5, 0.5),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.5, 0.5),
            pt2=(0.46, 0.5),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.23999998807907105, 0.5),
            pt2=(0.19999998807907104, 0.5),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.19999998807907104, 0.5),
            pt2=(0.19999998807907104, 0.44666666666666666),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
    ]
    assert len(test_annotation_helper.points) == 0


def test_multiple_detections():
    test_annotation_helper = TestAnnotationHelper()
    drawer = DetectionDrawer(test_annotation_helper, (1920, 1080))

    first_detection = create_detection(0.25, 0.25, 0.3, 0.3)
    second_detection = create_detection(0.75, 0.75, 0.3, 0.3)

    drawer.draw(first_detection)
    drawer.draw(second_detection)

    assert test_annotation_helper.rotated_rects == [
        RotatedRectData(
            center=dai.Point2f(0.25, 0.25),
            size=dai.Size2f(0.3, 0.3),
            angle=0.0,
            outline_color=PRIMARY_COLOR,
            fill_color=TRANSPARENT_PRIMARY_COLOR,
            thickness=0,
            clip_to_viewport=True,
        ),
        RotatedRectData(
            center=dai.Point2f(0.75, 0.75),
            size=dai.Size2f(0.3, 0.3),
            angle=0.0,
            outline_color=PRIMARY_COLOR,
            fill_color=TRANSPARENT_PRIMARY_COLOR,
            thickness=0,
            clip_to_viewport=True,
        ),
    ]
    assert test_annotation_helper.lines == [
        LineData(
            pt1=(0.09999999403953552, 0.17111110515064665),
            pt2=(0.09999999403953552, 0.09999999403953552),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.09999999403953552, 0.09999999403953552),
            pt2=(0.13999999403953553, 0.09999999403953552),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.3600000059604645, 0.09999999403953552),
            pt2=(0.4000000059604645, 0.09999999403953552),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.4000000059604645, 0.09999999403953552),
            pt2=(0.4000000059604645, 0.17111110515064665),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.4000000059604645, 0.32888889484935335),
            pt2=(0.4000000059604645, 0.4000000059604645),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.4000000059604645, 0.4000000059604645),
            pt2=(0.3600000059604645, 0.4000000059604645),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.13999999403953553, 0.4000000059604645),
            pt2=(0.09999999403953552, 0.4000000059604645),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.09999999403953552, 0.4000000059604645),
            pt2=(0.09999999403953552, 0.32888889484935335),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6000000238418579, 0.671111134952969),
            pt2=(0.6000000238418579, 0.6000000238418579),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6000000238418579, 0.6000000238418579),
            pt2=(0.640000023841858, 0.6000000238418579),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.859999976158142, 0.6000000238418579),
            pt2=(0.8999999761581421, 0.6000000238418579),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.8999999761581421, 0.6000000238418579),
            pt2=(0.8999999761581421, 0.671111134952969),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.8999999761581421, 0.828888865047031),
            pt2=(0.8999999761581421, 0.8999999761581421),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.8999999761581421, 0.8999999761581421),
            pt2=(0.859999976158142, 0.8999999761581421),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.640000023841858, 0.8999999761581421),
            pt2=(0.6000000238418579, 0.8999999761581421),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6000000238418579, 0.8999999761581421),
            pt2=(0.6000000238418579, 0.828888865047031),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=10.0,
            clip_to_viewport=True,
        ),
    ]


def test_zero_size_detection():
    test_annotation_helper = TestAnnotationHelper()
    drawer = DetectionDrawer(test_annotation_helper, (800, 600))

    detection = create_detection(0.5, 0.5, 0.0, 0.0)
    drawer.draw(detection)

    assert test_annotation_helper.rotated_rects == [
        RotatedRectData(
            center=dai.Point2f(0.5, 0.5),
            size=dai.Size2f(0.0, 0.0),
            angle=0.0,
            outline_color=PRIMARY_COLOR,
            fill_color=TRANSPARENT_PRIMARY_COLOR,
            thickness=0,
            clip_to_viewport=True,
        )
    ]
    line = LineData(
        pt1=(0.5, 0.5),
        pt2=(0.5, 0.5),
        color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
        thickness=2.3333333333333335,
        clip_to_viewport=True,
    )
    assert test_annotation_helper.lines == [line] * 8


def test_corner_lines_thickness():
    test_annotation_helper = TestAnnotationHelper()
    drawer = DetectionDrawer(test_annotation_helper, (800, 600))

    detection = create_detection(0.5, 0.5, 0.2, 0.15)
    drawer.draw(detection)

    assert test_annotation_helper.lines == [
        LineData(
            pt1=(0.4000000059604645, 0.4783333452542623),
            pt2=(0.4000000059604645, 0.42500001192092896),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.4000000059604645, 0.42500001192092896),
            pt2=(0.44000000596046446, 0.42500001192092896),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.5600000238418579, 0.42500001192092896),
            pt2=(0.6000000238418579, 0.42500001192092896),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6000000238418579, 0.42500001192092896),
            pt2=(0.6000000238418579, 0.4783333452542623),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6000000238418579, 0.5216666547457377),
            pt2=(0.6000000238418579, 0.574999988079071),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.6000000238418579, 0.574999988079071),
            pt2=(0.5600000238418579, 0.574999988079071),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.44000000596046446, 0.574999988079071),
            pt2=(0.4000000059604645, 0.574999988079071),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
        LineData(
            pt1=(0.4000000059604645, 0.574999988079071),
            pt2=(0.4000000059604645, 0.5216666547457377),
            color=(0.08235294371843338, 0.49803921580314636, 0.3450980484485626, 1.0),
            thickness=4.666666666666667,
            clip_to_viewport=True,
        ),
    ]
