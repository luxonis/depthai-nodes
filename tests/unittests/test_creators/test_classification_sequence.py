import numpy as np
import pytest

from depthai_nodes import Classifications
from depthai_nodes.message.creators import (
    create_classification_sequence_message,
)

CLASSES = ["cat", "dog", "bird"]
SCORES = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]]
CORRECT_SCORES = [0.7, 0.8, 0.5]
PROBS = [
    [0.5, 0.1, 0.1, 0.1, 0.2],
    [0.1, 0.5, 0.1, 0.1, 0.2],
    [0.1, 0.1, 0.5, 0.1, 0.2],
    [0.1, 0.1, 0.1, 0.5, 0.2],
    [0.2, 0.1, 0.1, 0.1, 0.5],
]


def test_valid_input():
    message = create_classification_sequence_message(CLASSES, SCORES)

    assert isinstance(message, Classifications)
    assert message.classes == CLASSES
    assert np.array_equal(message.scores, np.array(CORRECT_SCORES, dtype=np.float32))


def test_none_classes():
    with pytest.raises(ValueError):
        create_classification_sequence_message(None, SCORES)


def test_empty_scores():
    with pytest.raises(ValueError):
        create_classification_sequence_message(CLASSES, [[]])


def test_invalid_classes():
    with pytest.raises(ValueError):
        create_classification_sequence_message("not a list", SCORES)


def test_1d_scores():
    with pytest.raises(ValueError):
        create_classification_sequence_message(CLASSES, [0.5, 0.2, 0.3])


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        create_classification_sequence_message(CLASSES, [[0.7, 0.2], [0.1, 0.8]])


def test_scores_out_of_range():
    with pytest.raises(ValueError):
        create_classification_sequence_message(CLASSES, [[1.2, 0.2, 0.1]])


def test_scores_not_sum_to_1():
    with pytest.raises(ValueError):
        create_classification_sequence_message(CLASSES, [[0.7, 0.2, 0.2]])


def test_invalid_ignored_indexes():
    with pytest.raises(ValueError):
        create_classification_sequence_message(
            CLASSES, SCORES, ignored_indexes="not a list"
        )


def test_ignored_indexes_out_of_range():
    with pytest.raises(ValueError):
        create_classification_sequence_message(CLASSES, SCORES, ignored_indexes=[3])


def test_integer_ignored_indexes():
    with pytest.raises(ValueError):
        create_classification_sequence_message(CLASSES, SCORES, ignored_indexes=[1.0])


def test_2D_list_ignored_integers():
    with pytest.raises(ValueError):
        create_classification_sequence_message(CLASSES, SCORES, ignored_indexes=[[3]])


def test_remove_duplicates():
    res = create_classification_sequence_message(
        CLASSES,
        [[0.5, 0.2, 0.3], [0.5, 0.2, 0.3]],
        remove_duplicates=True,
    )
    assert res.classes == ["cat"]
    assert np.all(res.scores == np.array([0.5], dtype=np.float32))


def test_concatenate_chars_nospace():
    res = create_classification_sequence_message(
        ["c", "a", "t"],
        [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.1, 0.7]],
        concatenate_classes=True,
    )
    assert res.classes == ["cat"]
    assert np.allclose(res.scores, [0.5666666])


def test_concatenate_chars_space():
    res = create_classification_sequence_message(
        ["c", "a", "t", " ", "d"], PROBS, concatenate_classes=True
    )
    assert res.classes == ["cat", "d"]
    assert np.allclose(res.scores, [0.5, 0.5])


def test_concatenate_words():
    res = create_classification_sequence_message(
        ["Quick", "brown", "fox", "jumps", "over"], PROBS, concatenate_classes=True
    )
    assert res.classes == ["Quick brown fox jumps over"]
    assert np.allclose(res.scores, [0.5])


def test_concatenate_words_ignore_first():
    res = create_classification_sequence_message(
        ["Slow", "Quick", "brown", "fox", "jumps"],
        PROBS,
        concatenate_classes=True,
        ignored_indexes=[0],
    )
    assert res.classes == ["Quick brown fox jumps"]
    assert np.allclose(res.scores, [0.5])


def test_concatenate_mixed_words():
    res = create_classification_sequence_message(
        ["Quick", "b", "fox", "jumps", "o"], PROBS, concatenate_classes=True
    )
    assert res.classes == ["Quick b fox jumps o"]
    assert np.allclose(res.scores, [0.5])
