import re

import numpy as np
import pytest

from depthai_nodes.ml.messages.creators import (
    create_classification_sequence_message,
)


def test_none_classes():
    with pytest.raises(ValueError):
        create_classification_sequence_message(None, [0.5, 0.2, 0.3])


def test_1D_scores():
    with pytest.raises(
        ValueError, match=re.escape("Scores should be a 2D array, got (3,).")
    ):
        create_classification_sequence_message(["cat", "dog", "bird"], [0.5, 0.2, 0.3])


def test_empty_scores():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of classes and scores mismatch. Provided 3 class names and 0 scores."
        ),
    ):
        create_classification_sequence_message(["cat", "dog", "bird"], [[]])


def test_scores():
    with pytest.raises(
        ValueError, match=re.escape("Scores should be in the range [0, 1].")
    ):
        create_classification_sequence_message(
            ["cat", "dog", "bird"], [[0.5, -0.2, 1.3]]
        )


def test_probabilities():
    with pytest.raises(
        ValueError, match=re.escape("Each row of scores should sum to 1.")
    ):
        create_classification_sequence_message(
            ["cat", "dog", "bird"], [[0.5, 0.2, 0.3], [0.5, 0.2, 0.4]]
        )


def test_non_list_ignored_indexes():
    with pytest.raises(
        ValueError,
        match=re.escape("Ignored indexes should be a list, got <class 'float'>."),
    ):
        create_classification_sequence_message(
            ["cat", "dog", "bird"], [[0.5, 0.2, 0.3]], ignored_indexes=1.0
        )


def test_integer_ignored_indexes():
    with pytest.raises(
        ValueError, match=re.escape("Ignored indexes should be integers.")
    ):
        create_classification_sequence_message(
            ["cat", "dog", "bird"], [[0.5, 0.2, 0.3]], ignored_indexes=[1.0]
        )


def test_2D_list_integers():
    with pytest.raises(
        ValueError, match=re.escape("Ignored indexes should be integers.")
    ):
        create_classification_sequence_message(
            ["cat", "dog", "bird"], [[0.5, 0.2, 0.3]], ignored_indexes=[[3]]
        )


def test_upper_limit_integers():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Ignored indexes should be integers in the range [0, num_classes -1]."
        ),
    ):
        create_classification_sequence_message(
            ["cat", "dog", "bird"], [[0.5, 0.2, 0.3]], ignored_indexes=[3]
        )


def test_lower_limit_integers():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Ignored indexes should be integers in the range [0, num_classes -1]."
        ),
    ):
        create_classification_sequence_message(
            ["cat", "dog", "bird"], [[0.5, 0.2, 0.3]], ignored_indexes=[-1]
        )


def test_remove_duplicates():
    res = create_classification_sequence_message(
        ["cat", "dog", "bird"],
        [[0.5, 0.2, 0.3], [0.5, 0.2, 0.3]],
        remove_duplicates=True,
    )
    assert res.classes == ["cat"]
    assert np.all(res.scores == np.array([0.5], dtype=np.float32))


def test_ignored_indexes():
    res = create_classification_sequence_message(
        ["cat", "dog", "bird"], [[0.5, 0.2, 0.3], [0.1, 0.6, 0.3]], ignored_indexes=[1]
    )
    assert res.classes == ["cat"]
    assert np.all(res.scores == np.array([0.5], dtype=np.float32))


def test_all_ignored_indexes():
    res = create_classification_sequence_message(
        ["cat", "dog", "bird"],
        [[0.5, 0.2, 0.3], [0.1, 0.6, 0.3]],
        ignored_indexes=[0, 1, 2],
    )
    assert len(res.classes) == 0
    assert len(res.scores) == 0


def test_two_ignored_indexes():
    res = create_classification_sequence_message(
        ["cat", "dog", "bird"],
        [[0.5, 0.2, 0.3], [0.1, 0.6, 0.3]],
        ignored_indexes=[0, 2],
    )
    assert res.classes == ["dog"]
    assert np.all(res.scores == np.array([0.6], dtype=np.float32))


def test_concatenate_chars_nospace():
    res = create_classification_sequence_message(
        ["c", "a", "t"],
        [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.1, 0.7]],
        concatenate_classes=True,
    )
    assert res.classes == ["cat"]
    assert np.allclose(res.scores, [0.5666666])


def test_concatenate_chars_space():
    probs = [
        [0.5, 0.1, 0.1, 0.1, 0.2],
        [0.1, 0.5, 0.1, 0.1, 0.2],
        [0.1, 0.1, 0.5, 0.1, 0.2],
        [0.1, 0.1, 0.1, 0.5, 0.2],
        [0.2, 0.1, 0.1, 0.1, 0.5],
    ]
    res = create_classification_sequence_message(
        ["c", "a", "t", " ", "d"], probs, concatenate_classes=True
    )
    assert res.classes == ["cat", "d"]
    assert np.allclose(res.scores, [0.5, 0.5])


def test_concatenate_multiple_spaces():
    probs = [
        [0.5, 0.1, 0.1, 0.1, 0.2],
        [0.1, 0.5, 0.1, 0.1, 0.2],
        [0.1, 0.1, 0.5, 0.1, 0.2],
        [0.1, 0.1, 0.1, 0.5, 0.2],
        [0.1, 0.1, 0.1, 0.5, 0.2],
        [0.2, 0.1, 0.1, 0.1, 0.5],
        [0.1, 0.1, 0.1, 0.5, 0.2],
    ]
    res = create_classification_sequence_message(
        ["c", "a", "t", " ", "d"], probs, concatenate_classes=True
    )
    assert res.classes == ["cat", "d"]
    assert np.allclose(res.scores, [0.5, 0.5])


def test_concatenate_words():
    probs = [
        [0.5, 0.1, 0.1, 0.1, 0.2],
        [0.1, 0.5, 0.1, 0.1, 0.2],
        [0.1, 0.1, 0.5, 0.1, 0.2],
        [0.1, 0.1, 0.1, 0.5, 0.2],
        [0.2, 0.1, 0.1, 0.1, 0.5],
    ]

    res = create_classification_sequence_message(
        ["Quick", "brown", "fox", "jumps", "over"], probs, concatenate_classes=True
    )
    assert res.classes == ["Quick brown fox jumps over"]
    assert np.allclose(res.scores, [0.5])


def test_concatenate_words_ignore_first():
    probs = [
        [0.5, 0.1, 0.1, 0.1, 0.2],
        [0.1, 0.5, 0.1, 0.1, 0.2],
        [0.1, 0.1, 0.5, 0.1, 0.2],
        [0.1, 0.1, 0.1, 0.5, 0.2],
        [0.2, 0.1, 0.1, 0.1, 0.5],
    ]

    res = create_classification_sequence_message(
        ["Slow", "Quick", "brown", "fox", "jumps"],
        probs,
        concatenate_classes=True,
        ignored_indexes=[0],
    )
    assert res.classes == ["Quick brown fox jumps"]
    assert np.allclose(res.scores, [0.5])


def test_concatenate_mixed_words():
    probs = [
        [0.5, 0.1, 0.1, 0.1, 0.2],
        [0.1, 0.5, 0.1, 0.1, 0.2],
        [0.1, 0.1, 0.5, 0.1, 0.2],
        [0.1, 0.1, 0.1, 0.5, 0.2],
        [0.2, 0.1, 0.1, 0.1, 0.5],
    ]

    res = create_classification_sequence_message(
        ["Quick", "b", "fox", "jumps", "o"], probs, concatenate_classes=True
    )
    assert res.classes == ["Quick b fox jumps o"]
    assert np.allclose(res.scores, [0.5])


if __name__ == "__main__":
    pytest.main()
