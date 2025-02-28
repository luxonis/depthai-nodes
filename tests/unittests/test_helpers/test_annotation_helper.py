from datetime import timedelta

import depthai as dai

from depthai_nodes.utils.annotation_helper import AnnotationHelper


def test_empty_build():
    annotation_helper = AnnotationHelper()
    annots = annotation_helper.build(timedelta(), 0)
    assert len(annots.annotations[0].points) == 0
    assert len(annots.annotations[0].circles) == 0
    assert len(annots.annotations[0].texts) == 0


def test_draw_line():
    annotation_helper = AnnotationHelper()
    annotation_helper.draw_line((0.0, 0.0), (1.0, 1.0), (1.0, 0.0, 0.0, 1.0), 2)
    annots = annotation_helper.build(timedelta(), 0)

    assert [(i.x, i.y) for i in annots.annotations[0].points[0].points] == [
        (0.0, 0.0),
        (1.0, 1.0),
    ]
    assert annots.annotations[0].points[0].thickness == 2
    assert annots.annotations[0].points[0].type == dai.PointsAnnotationType.LINE_STRIP
    assert annots.annotations[0].points[0].outlineColor.r == 1.0
    assert annots.annotations[0].points[0].outlineColor.g == 0.0
    assert annots.annotations[0].points[0].outlineColor.b == 0.0
    assert annots.annotations[0].points[0].outlineColor.a == 1.0


def test_draw_polyline():
    annotation_helper = AnnotationHelper()
    annotation_helper.draw_polyline(
        [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        2,
        True,
    )
    annots = annotation_helper.build(timedelta(), 0)

    assert [(i.x, i.y) for i in annots.annotations[0].points[0].points] == [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ]
    assert annots.annotations[0].points[0].thickness == 2
    assert annots.annotations[0].points[0].type == dai.PointsAnnotationType.LINE_LOOP
    assert annots.annotations[0].points[0].outlineColor.r == 1.0
    assert annots.annotations[0].points[0].outlineColor.g == 0.0
    assert annots.annotations[0].points[0].outlineColor.b == 0.0
    assert annots.annotations[0].points[0].outlineColor.a == 1.0
    assert annots.annotations[0].points[0].fillColor.r == 0.0
    assert annots.annotations[0].points[0].fillColor.g == 1.0
    assert annots.annotations[0].points[0].fillColor.b == 0.0
    assert annots.annotations[0].points[0].fillColor.a == 1.0


def test_draw_polyline_closed():
    annotation_helper = AnnotationHelper()
    annotation_helper.draw_polyline(
        [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
        (1.0, 0.0, 0.0, 1.0),
        None,
        2,
        False,
    )
    annots = annotation_helper.build(timedelta(), 0)

    assert [(i.x, i.y) for i in annots.annotations[0].points[0].points] == [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ]
    assert annots.annotations[0].points[0].thickness == 2
    assert annots.annotations[0].points[0].type == dai.PointsAnnotationType.LINE_STRIP
    assert annots.annotations[0].points[0].outlineColor.r == 1.0
    assert annots.annotations[0].points[0].outlineColor.g == 0.0
    assert annots.annotations[0].points[0].outlineColor.b == 0.0
    assert annots.annotations[0].points[0].outlineColor.a == 1.0
    assert annots.annotations[0].points[0].fillColor.r == 0.0
    assert annots.annotations[0].points[0].fillColor.g == 0.0
    assert annots.annotations[0].points[0].fillColor.b == 0.0
    assert annots.annotations[0].points[0].fillColor.a == 0.0


def test_draw_circle():
    annotation_helper = AnnotationHelper()
    annotation_helper.draw_circle(
        (0.0, 0.0), 0.5, (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), 2
    )
    annots = annotation_helper.build(timedelta(), 0)

    assert annots.annotations[0].circles[0].position.x == 0.0
    assert annots.annotations[0].circles[0].position.y == 0.0
    assert annots.annotations[0].circles[0].diameter == 1.0
    assert annots.annotations[0].circles[0].outlineColor.r == 1.0
    assert annots.annotations[0].circles[0].outlineColor.g == 0.0
    assert annots.annotations[0].circles[0].outlineColor.b == 0.0
    assert annots.annotations[0].circles[0].outlineColor.a == 1.0
    assert annots.annotations[0].circles[0].fillColor.r == 0.0
    assert annots.annotations[0].circles[0].fillColor.g == 1.0
    assert annots.annotations[0].circles[0].fillColor.b == 0.0
    assert annots.annotations[0].circles[0].fillColor.a == 1.0
    assert annots.annotations[0].circles[0].thickness == 2


def test_draw_rectangle():
    annotation_helper = AnnotationHelper()
    annotation_helper.draw_rectangle(
        (0.0, 0.0), (1.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), 2
    )
    annots = annotation_helper.build(timedelta(), 0)

    assert [(i.x, i.y) for i in annots.annotations[0].points[0].points] == [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
    assert annots.annotations[0].points[0].outlineColor.r == 1.0
    assert annots.annotations[0].points[0].outlineColor.g == 0.0
    assert annots.annotations[0].points[0].outlineColor.b == 0.0
    assert annots.annotations[0].points[0].outlineColor.a == 1.0
    assert annots.annotations[0].points[0].fillColor.r == 0.0
    assert annots.annotations[0].points[0].fillColor.g == 1.0
    assert annots.annotations[0].points[0].fillColor.b == 0.0
    assert annots.annotations[0].points[0].fillColor.a == 1.0
    assert annots.annotations[0].points[0].thickness == 2
    assert annots.annotations[0].points[0].type == dai.PointsAnnotationType.LINE_LOOP


def test_draw_text():
    annotation_helper = AnnotationHelper()
    annotation_helper.draw_text(
        "Test", (0.0, 0.0), (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), 2
    )
    annots = annotation_helper.build(timedelta(), 0)

    assert annots.annotations[0].texts[0].text == "Test"
    assert annots.annotations[0].texts[0].position.x == 0.0
    assert annots.annotations[0].texts[0].position.y == 0.0
    assert annots.annotations[0].texts[0].backgroundColor.r == 0.0
    assert annots.annotations[0].texts[0].backgroundColor.g == 1.0
    assert annots.annotations[0].texts[0].backgroundColor.b == 0.0
    assert annots.annotations[0].texts[0].backgroundColor.a == 1.0
    assert annots.annotations[0].texts[0].textColor.r == 1.0
    assert annots.annotations[0].texts[0].textColor.g == 0.0
    assert annots.annotations[0].texts[0].textColor.b == 0.0
    assert annots.annotations[0].texts[0].textColor.a == 1.0
    assert annots.annotations[0].texts[0].fontSize == 2


def test_draw_points():
    annotation_helper = AnnotationHelper()
    annotation_helper.draw_points([(0.0, 0.0), (1.0, 1.0)], (1.0, 0.0, 0.0, 1.0), 2)
    annots = annotation_helper.build(timedelta(), 0)

    assert [(i.x, i.y) for i in annots.annotations[0].points[0].points] == [
        (0.0, 0.0),
        (1.0, 1.0),
    ]
    assert annots.annotations[0].points[0].thickness == 2
    assert annots.annotations[0].points[0].type == dai.PointsAnnotationType.POINTS
    assert annots.annotations[0].points[0].outlineColor.r == 1.0
    assert annots.annotations[0].points[0].outlineColor.g == 0.0
    assert annots.annotations[0].points[0].outlineColor.b == 0.0
    assert annots.annotations[0].points[0].outlineColor.a == 1.0
