import re

import numpy as np
import pytest

from depthai_nodes.ml.messages import Line, Lines
from depthai_nodes.ml.messages.creators.detection import create_line_detection_message


def test_not_numpy_lines():
    with pytest.raises(
        ValueError, match="Lines should be numpy array, got <class 'list'>."
    ):
        create_line_detection_message([1, 2, 3], [0.1, 0.2, 0.3])


def test_empty_numpy_lines():
    lines_msg = create_line_detection_message(np.array([]), np.array([]))
    assert isinstance(lines_msg, Lines)
    assert len(lines_msg.lines) == 0


def test_lines_shape():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Lines should be of shape (N,4) meaning [...,[x_start, y_start, x_end, y_end],...], got (1,)."
        ),
    ):
        create_line_detection_message(np.array([1]), [0.1])


def test_lines_dim_4():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Lines 2nd dimension should be of size 4 e.g. [x_start, y_start, x_end, y_end] got 3."
        ),
    ):
        create_line_detection_message(np.array([[1, 2, 3]]), [0.1])


def test_not_numpy_scores():
    with pytest.raises(
        ValueError, match="Scores should be numpy array, got <class 'list'>."
    ):
        create_line_detection_message(np.array([[1, 2, 3, 4]]), [0.1, 0.2, 0.3])


def test_empty_numpy_scores():
    with pytest.raises(
        ValueError, match="Scores should have same length as lines, got 0 and 1."
    ):
        create_line_detection_message(np.array([[1, 2, 3, 4]]), np.array([]))


def test_mixed_scores():
    with pytest.raises(
        ValueError,
        match="Scores should be of type float, got <class 'NoneType'>.",
    ):
        create_line_detection_message(
            np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), np.array([0.1, None])
        )


def test_scores_shape():
    with pytest.raises(
        ValueError,
        match=re.escape("Scores should be of shape (N,) meaning, got (1, 1)."),
    ):
        create_line_detection_message(np.array([[1, 2, 3, 4]]), np.array([[0.1]]))


def test_scores_length():
    with pytest.raises(
        ValueError,
        match=re.escape("Scores should have same length as lines, got 1 and 2."),
    ):
        create_line_detection_message(
            np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), np.array([0.1])
        )


def test_line_detection():
    lines = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    scores = np.array([0.4, 0.2])
    lines_msg = create_line_detection_message(lines, scores)

    assert isinstance(lines_msg, Lines)
    assert len(lines_msg.lines) == 2

    for i, line in enumerate(lines_msg.lines):
        assert isinstance(line, Line)
        assert line.start_point.x == lines[i][0]
        assert line.start_point.y == lines[i][1]
        assert line.end_point.x == lines[i][2]
        assert line.end_point.y == lines[i][3]
        assert line.confidence == scores[i]


if __name__ == "__main__":
    pytest.main()
