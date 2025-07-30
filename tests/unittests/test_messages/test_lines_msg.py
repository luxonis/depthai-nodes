import depthai as dai
import numpy as np
import pytest

from depthai_nodes import Line, Lines


@pytest.fixture
def line():
    return Line()


@pytest.fixture
def lines():
    return Lines()


def test_line_initialization(line: Line):
    assert line.start_point is None
    assert line.end_point is None
    assert line.confidence is None


def test_line_set_start_point(line: Line):
    start_point = dai.Point2f(0.1, 0.2)
    line.start_point = start_point
    assert np.allclose(line.start_point.x, 0.1, atol=1e-3)
    assert np.allclose(line.start_point.y, 0.2, atol=1e-3)

    with pytest.raises(TypeError):
        line.start_point = "not a Point2f"


def test_line_set_end_point(line: Line):
    end_point = dai.Point2f(0.3, 0.4)
    line.end_point = end_point
    assert np.allclose(line.end_point.x, 0.3, atol=1e-3)
    assert np.allclose(line.end_point.y, 0.4, atol=1e-3)

    with pytest.raises(TypeError):
        line.end_point = "not a Point2f"


def test_line_set_confidence(line: Line):
    line.confidence = 0.9
    assert line.confidence == 0.9

    line.confidence = 1.05
    assert line.confidence == 1.0
    assert isinstance(line.confidence, float)

    line.confidence = -0.05
    assert line.confidence == 0.0
    assert isinstance(line.confidence, float)

    with pytest.raises(TypeError):
        line.confidence = "not a float"

    with pytest.raises(ValueError):
        line.confidence = 1.5


def test_lines_initialization(lines: Lines):
    assert lines.lines == []
    assert lines.transformation is None


def test_lines_set_lines(lines: Lines):
    line1 = Line()
    line2 = Line()
    lines_list = [line1, line2]
    lines.lines = lines_list
    assert lines.lines == lines_list

    with pytest.raises(TypeError):
        lines.lines = "not a list"

    with pytest.raises(ValueError):
        lines.lines = [line1, "not a Line"]


def test_lines_set_transformation(lines: Lines):
    transformation = dai.ImgTransformation()
    lines.transformation = transformation
    assert lines.transformation == transformation

    with pytest.raises(TypeError):
        lines.transformation = "not a dai.ImgTransformation"


def test_lines_set_transformation_none(lines: Lines):
    lines.transformation = None
    assert lines.transformation is None
