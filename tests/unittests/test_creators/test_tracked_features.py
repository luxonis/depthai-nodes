import depthai as dai
import numpy as np
import pytest

from depthai_nodes.ml.messages.creators.tracked_features import (
    create_tracked_features_message,
)


def test_valid_input():
    reference_points = np.array([[0.1, 0.2], [0.3, 0.4]])
    target_points = np.array([[0.5, 0.6], [0.7, 0.8]])
    message = create_tracked_features_message(reference_points, target_points)

    assert isinstance(message, dai.TrackedFeatures)
    assert len(message.trackedFeatures) == 4
    assert all(
        isinstance(feature, dai.TrackedFeature) for feature in message.trackedFeatures
    )

    expected_values = [
        (0.1, 0.2, 0, 0),
        (0.5, 0.6, 0, 1),
        (0.3, 0.4, 1, 0),
        (0.7, 0.8, 1, 1),
    ]

    for feature, (expected_x, expected_y, expected_id, expected_age) in zip(
        message.trackedFeatures, expected_values
    ):
        assert np.allclose(feature.position.x, expected_x, atol=1e-3)
        assert np.allclose(feature.position.y, expected_y, atol=1e-3)
        assert np.allclose(feature.id, expected_id, atol=1e-3)
        assert np.allclose(feature.age, expected_age, atol=1e-3)


def test_invalid_reference_points_type():
    target_points = np.array([[0.5, 0.6], [0.7, 0.8]])
    with pytest.raises(
        ValueError, match="reference_points should be numpy array, got <class 'list'>."
    ):
        create_tracked_features_message([[0.1, 0.2], [0.3, 0.4]], target_points)


def test_invalid_reference_points_shape():
    reference_points = np.array([0.1, 0.2, 0.3, 0.4])
    target_points = np.array([[0.5, 0.6], [0.7, 0.8]])
    with pytest.raises(ValueError, match="reference_points should be of shape"):
        create_tracked_features_message(reference_points, target_points)


def test_invalid_reference_points_dimension():
    reference_points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    target_points = np.array([[0.5, 0.6], [0.7, 0.8]])
    with pytest.raises(
        ValueError, match="reference_points 2nd dimension should be of size 2"
    ):
        create_tracked_features_message(reference_points, target_points)


def test_invalid_target_points_type():
    reference_points = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(
        ValueError, match="target_points should be numpy array, got <class 'list'>."
    ):
        create_tracked_features_message(reference_points, [[0.5, 0.6], [0.7, 0.8]])


def test_invalid_target_points_shape():
    reference_points = np.array([[0.1, 0.2], [0.3, 0.4]])
    target_points = np.array([0.5, 0.6, 0.7, 0.8])
    with pytest.raises(ValueError, match="target_points should be of shape"):
        create_tracked_features_message(reference_points, target_points)


def test_invalid_target_points_dimension():
    reference_points = np.array([[0.1, 0.2], [0.3, 0.4]])
    target_points = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
    with pytest.raises(
        ValueError, match="target_points 2nd dimension should be of size 2"
    ):
        create_tracked_features_message(reference_points, target_points)


def test_mismatched_points_length():
    reference_points = np.array([[0.1, 0.2], [0.3, 0.4]])
    target_points = np.array([[0.5, 0.6]])
    with pytest.raises(
        ValueError,
        match="The number of reference points and target points should be the same.",
    ):
        create_tracked_features_message(reference_points, target_points)
