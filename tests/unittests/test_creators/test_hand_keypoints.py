import depthai as dai
import numpy as np
import pytest

from depthai_nodes.ml.messages import HandKeypoints
from depthai_nodes.ml.messages.creators.keypoints import create_hand_keypoints_message


def test_none_hand_keypoints():
    with pytest.raises(ValueError):
        create_hand_keypoints_message(None, 0.5, 0.8, 0.7)


def test_not_nparray_hand_keypoints():
    with pytest.raises(ValueError):
        create_hand_keypoints_message([[1, 2, 3]], 0.5, 0.8, 0.7)


def test_empty_nparray():
    with pytest.raises(ValueError):
        create_hand_keypoints_message(np.array([]), 0.5, 0.8, 0.7)


def test_3d_empty_nparray():
    with pytest.raises(ValueError):
        create_hand_keypoints_message(np.array([[[]]]), 0.5, 0.8, 0.7)


def test_wrong_dimension_nparray():
    with pytest.raises(ValueError):
        create_hand_keypoints_message(np.array([[], [], []]), 0.5, 0.8, 0.7)


def test_handedness_float():
    with pytest.raises(
        ValueError, match="Handedness should be float, got <class 'str'>."
    ):
        create_hand_keypoints_message(np.array([[1, 2, 3]]), "0.5", 0.8, 0.7)


def test_confidence_float():
    with pytest.raises(
        ValueError, match="Confidence should be float, got <class 'str'>."
    ):
        create_hand_keypoints_message(np.array([[1, 2, 3]]), 0.5, "0.8", 0.7)


def test_confidence_threshold_float():
    with pytest.raises(
        ValueError, match="Confidence threshold should be float, got <class 'str'>."
    ):
        create_hand_keypoints_message(np.array([[1, 2, 3]]), 0.5, 0.8, "0.7")


def test_low_confidence():
    create_hand_keypoints_message(np.array([[1, 2, 3]]), 0.5, 1e-10, 1e-11)


def test_negative_confidence():
    with pytest.raises(
        ValueError, match="Confidence should be between 0 and 1, got confidence -0.8."
    ):
        create_hand_keypoints_message(np.array([[1, 2, 3]]), 0.5, -0.8, 0.7)


def test_negative_confidence_threshold():
    with pytest.raises(
        ValueError,
        match="Confidence threshold should be between 0 and 1, got confidence threshold -0.7.",
    ):
        create_hand_keypoints_message(np.array([[1, 2, 3]]), 0.5, 0.8, -0.7)


def test_big_confidence():
    with pytest.raises(
        ValueError, match="Confidence should be between 0 and 1, got confidence 500.0."
    ):
        create_hand_keypoints_message(np.array([[1, 2, 3]]), 0.5, 500.0, 0.7)


def test_big_confidence_threshold():
    with pytest.raises(
        ValueError,
        match="Confidence threshold should be between 0 and 1, got confidence threshold 500.0.",
    ):
        create_hand_keypoints_message(np.array([[1, 2, 3]]), 0.5, 0.8, 500.0)


def test_handedness_between_0_and_1():
    with pytest.raises(ValueError):
        create_hand_keypoints_message(np.array([[1, 2, 3]]), 1.5, 0.8, 0.7)
    with pytest.raises(ValueError):
        create_hand_keypoints_message(np.array([[1, 2, 3]]), -0.5, 0.8, 0.7)


def test_confidence_below_thr():
    message = create_hand_keypoints_message(np.array([[1, 2, 3]]), 0.5, 0.8, 0.9)

    assert isinstance(message, HandKeypoints)
    assert message.confidence == 0.8
    assert message.handedness == 0.5
    assert message.keypoints == []


def test_confidence_above_thr():
    message = create_hand_keypoints_message(np.array([[1, 2, 3]]), 0.5, 0.8, 0.7)

    true_pt = dai.Point3f(1, 2, 3)
    true_pt.x = 1
    true_pt.y = 2
    true_pt.z = 3

    assert message.confidence == 0.8
    assert message.handedness == 0.5
    assert isinstance(message.keypoints, list)
    assert isinstance(message.keypoints[0], dai.Point3f)
    assert message.keypoints[0].x == true_pt.x
    assert message.keypoints[0].y == true_pt.y
    assert message.keypoints[0].z == true_pt.z


def test_list_keypoints():
    message = create_hand_keypoints_message(
        np.array([[1, 2, 3], [4, 5, 6]]), 0.5, 0.8, 0.7
    )

    assert len(message.keypoints) == 2
    assert message.keypoints[0].x == 1
    assert message.keypoints[0].y == 2
    assert message.keypoints[0].z == 3
    assert message.keypoints[1].x == 4
    assert message.keypoints[1].y == 5
    assert message.keypoints[1].z == 6


if __name__ == "__main__":
    pytest.main()
