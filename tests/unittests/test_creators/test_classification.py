import pytest

from depthai_nodes.ml.messages import Classifications
from depthai_nodes.ml.messages.creators.classification import (
    create_classification_message,
)


def test_none_classe():
    with pytest.raises(ValueError):
        create_classification_message(None, [0.5, 0.2, 0.3])


def test_none_scores():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], None)


def test_none_both():
    with pytest.raises(ValueError):
        create_classification_message(None, None)


def test_empty_classes():
    with pytest.raises(ValueError):
        create_classification_message([], [0.5, 0.2, 0.3])


def test_empty_scores():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], [])


def test_non_list_classes():
    with pytest.raises(ValueError):
        create_classification_message("cat", [0.5, 0.2, 0.3])


def test_non_np_scores():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], [int(1), int(2), int(3)])


def test_tuple_scores():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], (0.5, 0.2, 0.3))


def test_nd_scores():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], [[0.5, 0.2, 0.3]])


def test_mixed_scores():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], [0.5, 0.2, "30"])


def test_non_probability_scores():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], [0.2, 0.3, 0.4])


def test_non_probability_scores_2():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], [0.5, 0.5, 0.5])


def test_sum_above_upper_thr():  # upper thr is 1.01001001
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], [0.5, 0.11001001, 0.4])


def test_sum_below_upper_thr():
    create_classification_message(["cat", "dog", "bird"], [0.5, 0.11001000, 0.4])


def test_sum_above_lower_thr():  # lower thr is 0.98999001
    create_classification_message(["cat", "dog", "bird"], [0.5, 0.18999001, 0.3])


def test_sum_below_bottom_thr():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], [0.5, 0.18999, 0.3])


def test_mismatch_lengths():
    with pytest.raises(ValueError):
        create_classification_message(["cat", "dog", "bird"], [0.5, 0.2, 0.3, 0.4])


def test_correct_input():
    classes = ["cat", "dog", "bird"]
    scores = [0.2, 0.5, 0.3]

    correct_classes = ["dog", "bird", "cat"]
    correct_scores = [0.5, 0.3, 0.2]

    message = create_classification_message(classes, scores)

    assert isinstance(message, Classifications)
    assert message.classes == correct_classes
    assert message.scores == correct_scores
    assert isinstance(message.classes, list)
    assert isinstance(message.scores, list)


def test_single_class_and_score():
    classes = ["cat"]
    scores = [1.0]

    message = create_classification_message(classes, scores)
    assert message.classes == ["cat"]
    assert message.scores == [1.0]


def test_correct_input_with_mixed_classes():
    classes = ["cat", 1, None]
    scores = [0.2, 0.5, 0.3]

    correct_classes = [1, None, "cat"]
    correct_scores = [0.5, 0.3, 0.2]

    message = create_classification_message(classes, scores)

    assert isinstance(message, Classifications)
    assert message.classes == correct_classes
    assert message.scores == correct_scores
    assert isinstance(message.classes, list)
    assert isinstance(message.scores, list)


def test_very_small_scores():
    classes = ["cat", "dog", "bird"]
    scores = [1e-10, 1e-10, 1 - 2e-10]

    message = create_classification_message(classes, scores)

    assert isinstance(message, Classifications)
    assert message.classes == ["bird", "cat", "dog"]
    assert message.scores == [1 - 2e-10, 1e-10, 1e-10]


def test_identical_scores():
    classes = ["cat", "dog", "bird"]
    scores = [1 / 3, 1 / 3, 1 / 3]

    message = create_classification_message(classes, scores)

    assert message.classes == classes
    assert message.scores == scores


def test_duplicate_scores():
    classes = ["cat", "dog", "bird"]
    scores = [0.4, 0.2, 0.4]

    correct_classes = ["cat", "bird", "dog"]
    correct_scores = [0.4, 0.4, 0.2]

    message = create_classification_message(classes, scores)

    assert message.classes == correct_classes
    assert message.scores == correct_scores


if __name__ == "__main__":
    pytest.main()
