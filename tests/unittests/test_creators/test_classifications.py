import numpy as np
import pytest

from depthai_nodes import Classifications
from depthai_nodes.message.creators import (
    create_classification_message,
)

CLASSES = ["cat", "dog", "bird"]
SCORES = [0.7, 0.2, 0.1]


def test_valid_input():
    message = create_classification_message(CLASSES, SCORES)

    assert isinstance(message, Classifications)
    assert message.classes == ["cat", "dog", "bird"]
    assert np.array_equal(message.scores, np.array(SCORES, dtype=np.float32))


def test_single_class_and_score():
    classes = ["cat"]
    scores = [1.0]

    message = create_classification_message(classes, scores)
    assert message.classes == ["cat"]
    assert message.scores == [1.0]


def test_identical_scores():
    scores = [1 / 3, 1 / 3, 1 / 3]

    message = create_classification_message(CLASSES, scores)

    assert message.classes == CLASSES
    assert np.all(message.scores == np.array(scores, dtype=np.float32))


def test_duplicate_scores():
    scores = [0.4, 0.2, 0.4]

    correct_classes = ["cat", "bird", "dog"]
    correct_scores = [0.4, 0.4, 0.2]

    message = create_classification_message(CLASSES, scores)

    assert message.classes == correct_classes
    assert np.all(message.scores == np.array(correct_scores, dtype=np.float32))


def test_very_small_scores():
    scores = [1e-10, 1e-10, 1 - 2e-10]

    message = create_classification_message(CLASSES, scores)

    assert isinstance(message, Classifications)
    assert message.classes == ["bird", "cat", "dog"]
    assert np.all(
        message.scores == np.array([1 - 2e-10, 1e-10, 1e-10], dtype=np.float32)
    )


def test_none_classes():
    with pytest.raises(ValueError):
        create_classification_message(None, SCORES)


def test_non_list_classes():
    with pytest.raises(ValueError):
        create_classification_message("not a list", SCORES)


def test_empty_classes():
    with pytest.raises(ValueError):
        create_classification_message([], SCORES)


def test_none_scores():
    with pytest.raises(ValueError):
        create_classification_message(CLASSES, None)


def test_non_list_non_array_scores():
    with pytest.raises(ValueError):
        create_classification_message(CLASSES, "not a list or array")


def test_empty_scores():
    with pytest.raises(ValueError):
        create_classification_message(CLASSES, [])


def test_non_1d_array_scores():
    with pytest.raises(ValueError):
        create_classification_message(CLASSES, np.array([SCORES]))


def test_non_float_scores():
    with pytest.raises(ValueError):
        create_classification_message(CLASSES, [0.7, 0.2, "not a float"])


def test_scores_not_sum_to_1():
    with pytest.raises(ValueError):
        create_classification_message(CLASSES, [0.2, 0.2, 0.2])


def test_mismatched_classes_scores():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog"], [0.7, 0.2, 0.1])
