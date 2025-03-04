import pytest

from depthai_nodes import Keypoint, Keypoints
from depthai_nodes.message.creators import create_keypoints_message

KPTS = [[0.1, 0.2], [0.3, 0.4]]
SCORES = [0.9, 0.8]


def test_valid_2d_keypoints():
    message = create_keypoints_message(KPTS, SCORES)

    assert isinstance(message, Keypoints)
    assert len(message.keypoints) == 2
    assert all(isinstance(kp, Keypoint) for kp in message.keypoints)
    for i, kp in enumerate(message.keypoints):
        assert kp.x == KPTS[i][0]
        assert kp.y == KPTS[i][1]
        assert kp.z == 0.0
        assert kp.confidence == SCORES[i]


def test_valid_3d_keypoints():
    keypoints = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    scores = [0.9, 0.8]
    message = create_keypoints_message(keypoints, scores)

    assert isinstance(message, Keypoints)
    assert len(message.keypoints) == 2
    assert all(isinstance(kp, Keypoint) for kp in message.keypoints)
    for i, kp in enumerate(message.keypoints):
        assert kp.x == keypoints[i][0]
        assert kp.y == keypoints[i][1]
        assert kp.z == keypoints[i][2]
        assert kp.confidence == scores[i]


def test_valid_keypoints_no_scores():
    message = create_keypoints_message(KPTS)

    assert isinstance(message, Keypoints)
    assert len(message.keypoints) == 2
    assert all(isinstance(kp, Keypoint) for kp in message.keypoints)
    for kp in message.keypoints:
        assert kp.confidence == -1


def test_invalid_keypoints_type():
    with pytest.raises(ValueError):
        create_keypoints_message("not a list or array")


def test_invalid_scores_type():
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, scores="not a list or array")


def test_mismatched_keypoints_scores_length():
    scores = [0.9]
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, scores)


def test_invalid_scores_values():
    scores = [0.9, "not a float"]
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, scores)


def test_scores_out_of_range():
    scores = [0.9, 1.1]
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, scores)


def test_invalid_confidence_threshold_type():
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, SCORES, confidence_threshold="not a float")


def test_confidence_threshold_out_of_range():
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, SCORES, confidence_threshold=1.1)


def test_invalid_keypoints_shape():
    keypoints = [[0.1, 0.2, 0.3, 0.4]]
    with pytest.raises(ValueError):
        create_keypoints_message(keypoints)


def test_invalid_keypoints_inner_type():
    keypoints = [[0.1, "not a float"]]
    with pytest.raises(ValueError):
        create_keypoints_message(keypoints)


def test_all_keypoints_same_shape():
    keypoints = [[0.1, 0.2], [0.3, 0.4, 0.5]]
    with pytest.raises(ValueError):
        create_keypoints_message(keypoints)
