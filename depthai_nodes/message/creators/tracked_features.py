import depthai as dai
import numpy as np


def create_feature_point(x: float, y: float, id: int, age: int) -> dai.TrackedFeature:
    """Create a tracked feature point.

    Args:
        x: X coordinate of the feature point.
        y: Y coordinate of the feature point.
        id: ID of the feature point.
        age: Age of the feature point.

    Returns:
        Tracked feature point.
    """

    feature = dai.TrackedFeature()
    feature.position.x = x
    feature.position.y = y
    feature.id = id
    feature.age = age

    return feature


def create_tracked_features_message(
    reference_points: np.ndarray, target_points: np.ndarray
) -> dai.TrackedFeatures:
    """Create paired tracked features from reference and target points.

    Args:
        reference_points: NumPy array of shape ``(N, 2)`` with reference ``[x, y]``
            points.
        target_points: NumPy array of shape ``(N, 2)`` with matching target points.

    Returns:
        Message containing alternating reference and target features. Each pair shares
        an ID; reference features have age 0 and target features have age 1.

    Raises:
        ValueError: If either input is not a NumPy array of shape ``(N, 2)``, or the
            arrays contain different numbers of points.
    """

    if not isinstance(reference_points, np.ndarray):
        raise ValueError(
            f"reference_points should be numpy array, got {type(reference_points)}."
        )
    if len(reference_points.shape) != 2:
        raise ValueError(
            f"reference_points should be of shape (N,2) meaning [...,[x, y],...], got {reference_points.shape}."
        )
    if reference_points.shape[1] != 2:
        raise ValueError(
            f"reference_points 2nd dimension should be of size 2 e.g. [x, y], got {reference_points.shape[1]}."
        )
    if not isinstance(target_points, np.ndarray):
        raise ValueError(
            f"target_points should be numpy array, got {type(target_points)}."
        )
    if len(target_points.shape) != 2:
        raise ValueError(
            f"target_points should be of shape (N,2) meaning [...,[x, y],...], got {target_points.shape}."
        )
    if target_points.shape[1] != 2:
        raise ValueError(
            f"target_points 2nd dimension should be of size 2 e.g. [x, y], got {target_points.shape[1]}."
        )
    if len(reference_points) != len(target_points):
        raise ValueError(
            "The number of reference points and target points should be the same."
        )

    features = []

    for i in range(reference_points.shape[0]):
        reference_feature = create_feature_point(
            reference_points[i][0], reference_points[i][1], i, 0
        )
        target_feature = create_feature_point(
            target_points[i][0], target_points[i][1], i, 1
        )
        features.append(reference_feature)
        features.append(target_feature)

    features_msg = dai.TrackedFeatures()
    features_msg.trackedFeatures = features

    return features_msg
