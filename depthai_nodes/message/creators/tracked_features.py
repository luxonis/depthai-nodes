import depthai as dai
import numpy as np


def create_feature_point(x: float, y: float, id: int, age: int) -> dai.TrackedFeature:
    """Create a tracked feature point.

    @param x: X coordinate of the feature point.
    @type x: float
    @param y: Y coordinate of the feature point.
    @type y: float
    @param id: ID of the feature point.
    @type id: int
    @param age: Age of the feature point.
    @type age: int
    @return: Tracked feature point.
    @rtype: dai.TrackedFeature
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
    """Create a DepthAI message for tracked features.

    @param reference_points: Reference points of shape (N,2) meaning [...,[x, y],...].
    @type reference_points: np.ndarray
    @param target_points: Target points of shape (N,2) meaning [...,[x, y],...].
    @type target_points: np.ndarray

    @return: Message containing the tracked features.
    @rtype: dai.TrackedFeatures

    @raise ValueError: If the reference_points are not a numpy array.
    @raise ValueError: If the reference_points are not of shape (N,2).
    @raise ValueError: If the reference_points 2nd dimension is not of size E{2}.
    @raise ValueError: If the target_points are not a numpy array.
    @raise ValueError: If the target_points are not of shape (N,2).
    @raise ValueError: If the target_points 2nd dimension is not of size E{2}.
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
