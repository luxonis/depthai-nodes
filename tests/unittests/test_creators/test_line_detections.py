import numpy as np
import pytest

from depthai_nodes import Line, Lines
from depthai_nodes.message.creators import create_line_detection_message

LINE = np.array([[0.1, 0.2, 0.3, 0.4]])
SCORE = np.array([0.9])


def test_valid_input():
    lines = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    scores = np.array([0.9, 0.8])
    message = create_line_detection_message(lines, scores)

    assert isinstance(message, Lines)
    assert len(message.lines) == 2

    for i, line in enumerate(message.lines):
        assert isinstance(line, Line)
        assert np.allclose(line.start_point.x, lines[i][0], atol=1e-3)
        assert np.allclose(line.start_point.y, lines[i][1], atol=1e-3)
        assert np.allclose(line.end_point.x, lines[i][2], atol=1e-3)
        assert np.allclose(line.end_point.y, lines[i][3], atol=1e-3)
        assert np.allclose(line.confidence, scores[i], atol=1e-3)


def test_empty_lines():
    lines = np.array([])
    scores = np.array([])
    message = create_line_detection_message(lines, scores)

    assert isinstance(message, Lines)
    assert len(message.lines) == 0


def test_invalid_lines_type():
    with pytest.raises(ValueError):
        create_line_detection_message(LINE.tolist(), SCORE)


def test_invalid_lines_shape():
    with pytest.raises(ValueError):
        create_line_detection_message(LINE[0], SCORE)


def test_invalid_lines_dimension():
    with pytest.raises(ValueError):
        create_line_detection_message(LINE[:, :3], SCORE)


def test_invalid_scores_type():
    with pytest.raises(ValueError):
        create_line_detection_message(LINE, SCORE.tolist())


def test_invalid_scores_shape():
    with pytest.raises(ValueError):
        create_line_detection_message(LINE, np.array([SCORE]))


def test_invalid_scores_value_type():
    with pytest.raises(ValueError):
        create_line_detection_message(LINE, np.array([1], dtype=np.int64))


def test_mismatched_lines_scores_length():
    with pytest.raises(ValueError):
        create_line_detection_message(LINE, np.array([0.9, 0.8]))
