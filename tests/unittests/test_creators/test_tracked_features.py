import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message.creators import (
    create_tracked_features_message,
)

REFERENCE_POINTS = np.array([[0.1, 0.2], [0.3, 0.4]])
TARGET_POINTS = np.array([[0.5, 0.6], [0.7, 0.8]])


def test_valid_input():
    message = create_tracked_features_message(REFERENCE_POINTS, TARGET_POINTS)

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
    with pytest.raises(ValueError):
        create_tracked_features_message(REFERENCE_POINTS.tolist(), TARGET_POINTS)


def test_invalid_reference_points_shape():
    with pytest.raises(ValueError):
        create_tracked_features_message(REFERENCE_POINTS.flatten(), TARGET_POINTS)


def test_invalid_reference_points_dimension():
    reference_points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    with pytest.raises(ValueError):
        create_tracked_features_message(reference_points, TARGET_POINTS)


def test_invalid_target_points_type():
    with pytest.raises(ValueError):
        create_tracked_features_message(REFERENCE_POINTS, TARGET_POINTS.tolist())


def test_invalid_target_points_shape():
    with pytest.raises(ValueError):
        create_tracked_features_message(REFERENCE_POINTS, TARGET_POINTS.flatten())


def test_invalid_target_points_dimension():
    target_points = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
    with pytest.raises(ValueError):
        create_tracked_features_message(REFERENCE_POINTS, target_points)


def test_mismatched_points_length():
    target_points = np.array([[0.5, 0.6]])
    with pytest.raises(ValueError):
        create_tracked_features_message(REFERENCE_POINTS, target_points)
