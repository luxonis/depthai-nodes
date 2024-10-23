import re

import numpy as np
import pytest

from depthai_nodes.ml.messages import Keypoint, Keypoints
from depthai_nodes.ml.messages.creators.keypoints import create_keypoints_message


def test_keypoints_not_list():
    with pytest.raises(
        ValueError, match="Keypoints should be numpy array or list, got <class 'int'>."
    ):
        create_keypoints_message(keypoints=1)


def test_scores_not_list():
    with pytest.raises(
        ValueError, match="Scores should be numpy array or list, got <class 'int'>."
    ):
        create_keypoints_message(keypoints=np.array([[1.0, 2.0, 3.0]]), scores=1)


def test_keypoints_and_scores_length():
    with pytest.raises(
        ValueError,
        match="Keypoints and scores should have the same length. Got 1 keypoints and 2 scores.",
    ):
        create_keypoints_message(keypoints=np.array([[1.0, 2.0, 3.0]]), scores=[1, 2])


def test_non_float_scores():
    with pytest.raises(ValueError, match="Scores should only contain float values."):
        create_keypoints_message(keypoints=np.array([[1.0, 2.0, 3.0]]), scores=[1])


def test_higher_than_one_scores():
    with pytest.raises(ValueError, match="Scores should only contain values between 0 and 1."):
        create_keypoints_message(keypoints=np.array([[1.0, 2.0, 3.0]]), scores=[1.5])

def test_negative_scores():
    with pytest.raises(ValueError, match="Scores should only contain values between 0 and 1."):
        create_keypoints_message(keypoints=np.array([[1.0, 2.0, 3.0]]), scores=[-1.5])

def test_confidence_threshold_not_float():
    with pytest.raises(
        ValueError, match="The confidence_threshold should be float, got <class 'str'>."
    ):
        create_keypoints_message(keypoints=np.array([[1.0, 2.0, 3.0]]), scores=[0.5], confidence_threshold="0.8")


def test_confidence_threshold_between_0_and_1():
    with pytest.raises(
        ValueError,
        match="The confidence_threshold should be between 0 and 1, got confidence_threshold 1.5",
    ):
        create_keypoints_message(keypoints=np.array([[1.0, 2.0, 3.0]]), scores=[0.5], confidence_threshold=1.5)


def test_1d_keypoint():
    with pytest.raises(
        ValueError, match="All keypoints should be of dimension 2 or 3, got dimension 1"
    ):
        create_keypoints_message(keypoints=np.array([[1.0]]))


def test_consistent_keypoint_dims():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "All keypoints have to be of same dimension e.g. [x, y] or [x, y, z], got mixed inner dimensions."
        ),
    ):
        create_keypoints_message(
            keypoints=[[1.0, 2.0, 3.0], [1.0, 3.0], [1.0, 2.0, 3.0]]
        )


def test_nonlist_keypoint():
    with pytest.raises(
        ValueError,
        match="Keypoints should be list of lists or np.array, got list of <class 'str'>.",
    ):
        create_keypoints_message(
            keypoints=[[1.0, 2.0, 3.0],"1.0"]
        )


def test_nonfloat_keypoint():
    with pytest.raises(
        ValueError,
        match="Keypoints inner list should contain only float, got <class 'str'>.",
    ):
        create_keypoints_message(
            keypoints=[["1.0", 2.0, 3.0]]
        )


def test_keypoints_shape():
    with pytest.raises(
        ValueError,
        match=re.escape("Keypoints should be of shape (N,2 or 3) got (3, 2, 3)."),
    ):
        create_keypoints_message(
            np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                ]
            )
        )


def test_all_scores_below_thr():
    message = create_keypoints_message([[1.0, 2.0, 3.0]], [0.4], 0.6)

    assert isinstance(message, Keypoints)
    assert message.keypoints == []


def test_all_scores_above_thr():
    message = create_keypoints_message([[1.0, 2.0, 3.0]], [0.8], 0.6)

    true_pt = np.array([1.0, 2.0, 3.0])

    assert len(message.keypoints) == 1
    assert isinstance(message, Keypoints)
    assert isinstance(message.keypoints[0], Keypoint)
    assert message.keypoints[0].x == true_pt[0]
    assert message.keypoints[0].y == true_pt[1]
    assert message.keypoints[0].z == true_pt[2]


def test_2d_keypoints():
    message = create_keypoints_message([[1.0, 2.0], [3.0, 4.0]], [0.8, 0.5], 0.6)

    assert len(message.keypoints) == 1
    assert message.keypoints[0].x == 1.0
    assert message.keypoints[0].y == 2.0
    assert message.keypoints[0].z == 0.0


if __name__ == "__main__":
    pytest.main()
