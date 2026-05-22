import depthai as dai
import pytest

from depthai_nodes import Keypoints
from depthai_nodes.message.creators import create_keypoints_message

KPTS = [[0.1, 0.2], [0.3, 0.4]]
SCORES = [0.9, 0.8]


def test_valid_2d_keypoints():
    message = create_keypoints_message(KPTS, SCORES)

    assert isinstance(message, Keypoints)
    assert isinstance(message.keypoints_list, dai.KeypointsList)
    assert len(message.getKeypoints()) == 2
    for i, kp in enumerate(message.getKeypoints()):
        assert kp.imageCoordinates.x == pytest.approx(KPTS[i][0])
        assert kp.imageCoordinates.y == pytest.approx(KPTS[i][1])
        assert kp.imageCoordinates.z == pytest.approx(0.0)
        assert kp.confidence == pytest.approx(SCORES[i])


def test_valid_3d_keypoints():
    keypoints = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    scores = [0.9, 0.8]
    message = create_keypoints_message(keypoints, scores)

    assert isinstance(message, Keypoints)
    assert len(message.getKeypoints()) == 2
    for i, kp in enumerate(message.getKeypoints()):
        assert kp.imageCoordinates.x == pytest.approx(keypoints[i][0])
        assert kp.imageCoordinates.y == pytest.approx(keypoints[i][1])
        assert kp.imageCoordinates.z == pytest.approx(keypoints[i][2])
        assert kp.confidence == pytest.approx(scores[i])


def test_valid_keypoints_no_scores():
    message = create_keypoints_message(KPTS)

    assert isinstance(message, Keypoints)
    assert len(message.getKeypoints()) == 2
    for kp in message.getKeypoints():
        assert kp.confidence == -1


def test_invalid_keypoints_type():
    with pytest.raises(ValueError):
        create_keypoints_message("not a list or array")


def test_invalid_scores_type():
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, scores="not a list or array")


def test_mismatched_keypoints_scores_length():
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, [0.9])


def test_invalid_scores_values():
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, [0.9, "not a float"])


def test_scores_out_of_range():
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, [0.9, 1.1])


def test_invalid_confidence_threshold_type():
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, SCORES, confidence_threshold="not a float")


def test_confidence_threshold_out_of_range():
    with pytest.raises(ValueError):
        create_keypoints_message(KPTS, SCORES, confidence_threshold=1.1)


def test_invalid_keypoints_shape():
    with pytest.raises(ValueError):
        create_keypoints_message([[0.1, 0.2, 0.3, 0.4]])


def test_invalid_keypoints_inner_type():
    with pytest.raises(ValueError):
        create_keypoints_message([[0.1, "not a float"]])


def test_all_keypoints_same_shape():
    with pytest.raises(ValueError):
        create_keypoints_message([[0.1, 0.2], [0.3, 0.4, 0.5]])
