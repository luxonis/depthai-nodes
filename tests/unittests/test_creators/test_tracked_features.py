import re

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.ml.messages.creators.tracked_features import (
    create_tracked_features_message,
)


def test_reference_point_shape():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "reference_points should be of shape (N,2) meaning [...,[x, y],...], got (3,)."
        ),
    ):
        create_tracked_features_message(
            np.array([0.7, 0.3, 0.5]), target_points=np.array([[1, 2], [3, 4]])
        )


def test_reference_point_size():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "reference_points 2nd dimension should be of size 2 e.g. [x, y], got 1."
        ),
    ):
        create_tracked_features_message(
            np.array([[0.7], [0.8]]), target_points=np.array([[1, 2], [3, 4]])
        )


def test_target_point_shape():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "target_points should be of shape (N,2) meaning [...,[x, y],...], got (3,)."
        ),
    ):
        create_tracked_features_message(
            np.array([[0.7, 0.3], [0.5, 0.6]]), target_points=np.array([0.7, 0.3, 0.5])
        )


def test_target_point_size():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "target_points 2nd dimension should be of size 2 e.g. [x, y], got 1."
        ),
    ):
        create_tracked_features_message(
            np.array([[0.7, 0.3], [0.5, 0.6]]), target_points=np.array([[1], [2]])
        )


def test_same_length():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of reference points and target points should be the same."
        ),
    ):
        create_tracked_features_message(
            np.array([[0.7, 0.3], [0.5, 0.6], [0.5, 0.5], [0.6, 0.5]]),
            target_points=np.array([[1, 2], [3, 4]]),
        )


def test_return():
    reference_points = np.array([[0.7, 0.3], [0.5, 0.6], [0.5, 0.5], [0.6, 0.5]])
    target_points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    msg = create_tracked_features_message(reference_points, target_points)

    assert isinstance(msg, dai.TrackedFeatures)
    assert len(msg.trackedFeatures) == 8
    for i in range(4):
        assert isinstance(msg.trackedFeatures[2 * i], dai.TrackedFeature)
        assert np.isclose(msg.trackedFeatures[2 * i].position.x, reference_points[i][0])
        assert np.isclose(msg.trackedFeatures[2 * i].position.y, reference_points[i][1])
        assert msg.trackedFeatures[2 * i].id == i
        assert msg.trackedFeatures[2 * i].age == 0
        assert isinstance(msg.trackedFeatures[2 * i + 1], dai.TrackedFeature)
        assert np.isclose(
            msg.trackedFeatures[2 * i + 1].position.x, target_points[i][0]
        )
        assert np.isclose(
            msg.trackedFeatures[2 * i + 1].position.y, target_points[i][1]
        )
        assert msg.trackedFeatures[2 * i + 1].id == i
        assert msg.trackedFeatures[2 * i + 1].age == 1


if __name__ == "__main__":
    pytest.main()
