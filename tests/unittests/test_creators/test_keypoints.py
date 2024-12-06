import pytest

from depthai_nodes.ml.messages import Keypoint, Keypoints
from depthai_nodes.ml.messages.creators.keypoints import create_keypoints_message


def test_valid_2d_keypoints():
    keypoints = [[0.1, 0.2], [0.3, 0.4]]
    scores = [0.9, 0.8]
    message = create_keypoints_message(keypoints, scores)

    assert isinstance(message, Keypoints)
    assert len(message.keypoints) == 2
    assert all(isinstance(kp, Keypoint) for kp in message.keypoints)
    assert message.keypoints[0].x == 0.1
    assert message.keypoints[0].y == 0.2
    assert message.keypoints[0].z == 0.0
    assert message.keypoints[0].confidence == 0.9
    assert message.keypoints[1].x == 0.3
    assert message.keypoints[1].y == 0.4
    assert message.keypoints[1].z == 0.0
    assert message.keypoints[1].confidence == 0.8


def test_valid_3d_keypoints():
    keypoints = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    scores = [0.9, 0.8]
    message = create_keypoints_message(keypoints, scores)

    assert isinstance(message, Keypoints)
    assert len(message.keypoints) == 2
    assert all(isinstance(kp, Keypoint) for kp in message.keypoints)
    assert message.keypoints[0].x == 0.1
    assert message.keypoints[0].y == 0.2
    assert message.keypoints[0].z == 0.3
    assert message.keypoints[0].confidence == 0.9
    assert message.keypoints[1].x == 0.4
    assert message.keypoints[1].y == 0.5
    assert message.keypoints[1].z == 0.6
    assert message.keypoints[1].confidence == 0.8


def test_valid_keypoints_no_scores():
    keypoints = [[0.1, 0.2], [0.3, 0.4]]
    message = create_keypoints_message(keypoints)

    assert isinstance(message, Keypoints)
    assert len(message.keypoints) == 2
    assert all(isinstance(kp, Keypoint) for kp in message.keypoints)
    assert message.keypoints[0].x == 0.1
    assert message.keypoints[0].y == 0.2
    assert message.keypoints[0].z == 0.0
    assert message.keypoints[0].confidence == -1
    assert message.keypoints[1].x == 0.3
    assert message.keypoints[1].y == 0.4
    assert message.keypoints[1].z == 0.0
    assert message.keypoints[1].confidence == -1


def test_invalid_keypoints_type():
    with pytest.raises(
        ValueError, match="Keypoints should be numpy array or list, got <class 'str'>."
    ):
        create_keypoints_message("not a list or array")


def test_invalid_scores_type():
    keypoints = [[0.1, 0.2], [0.3, 0.4]]
    with pytest.raises(
        ValueError, match="Scores should be numpy array or list, got <class 'str'>."
    ):
        create_keypoints_message(keypoints, scores="not a list or array")


def test_mismatched_keypoints_scores_length():
    keypoints = [[0.1, 0.2], [0.3, 0.4]]
    scores = [0.9]
    with pytest.raises(
        ValueError, match="Keypoints and scores should have the same length."
    ):
        create_keypoints_message(keypoints, scores)


def test_invalid_scores_values():
    keypoints = [[0.1, 0.2], [0.3, 0.4]]
    scores = [0.9, "not a float"]
    with pytest.raises(ValueError, match="Scores should only contain float values."):
        create_keypoints_message(keypoints, scores)


def test_scores_out_of_range():
    keypoints = [[0.1, 0.2], [0.3, 0.4]]
    scores = [0.9, 1.1]
    with pytest.raises(
        ValueError, match="Scores should only contain values between 0 and 1."
    ):
        create_keypoints_message(keypoints, scores)


def test_invalid_confidence_threshold_type():
    keypoints = [[0.1, 0.2], [0.3, 0.4]]
    scores = [0.9, 0.8]
    with pytest.raises(
        ValueError, match="The confidence_threshold should be float, got <class 'str'>."
    ):
        create_keypoints_message(keypoints, scores, confidence_threshold="not a float")


def test_confidence_threshold_out_of_range():
    keypoints = [[0.1, 0.2], [0.3, 0.4]]
    scores = [0.9, 0.8]
    with pytest.raises(
        ValueError, match="The confidence_threshold should be between 0 and 1."
    ):
        create_keypoints_message(keypoints, scores, confidence_threshold=1.1)


def test_invalid_keypoints_shape():
    keypoints = [[0.1, 0.2, 0.3, 0.4]]
    with pytest.raises(
        ValueError,
        match="All keypoints should be of dimension 2 or 3, got dimension 4.",
    ):
        create_keypoints_message(keypoints)


def test_invalid_keypoints_inner_type():
    keypoints = [[0.1, "not a float"]]
    with pytest.raises(
        ValueError,
        match="Keypoints inner list should contain only float, got <class 'str'>.",
    ):
        create_keypoints_message(keypoints)
