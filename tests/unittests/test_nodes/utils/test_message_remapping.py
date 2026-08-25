from datetime import timedelta
from typing import Callable

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message.creators import (
    create_classification_message,
    create_cluster_message,
    create_detection_message,
    create_keypoints_message,
    create_line_detection_message,
    create_map_message,
    create_regression_message,
)
from depthai_nodes.node.coordinates_mapper import CoordinatesMapper
from depthai_nodes.node.utils.message_remapping import remap_message


def _create_segmentation_mask() -> dai.SegmentationMask:
    message = dai.SegmentationMask()
    message.setCvMask(np.array([[0, 1], [2, 255]], dtype=np.uint8))
    return message


MESSAGE_FACTORIES: list[Callable[[], dai.Buffer]] = [
    lambda: create_detection_message(
        bboxes=np.array([[0.2, 0.3, 0.4, 0.5]]),
        scores=np.array([0.9]),
    ),
    _create_segmentation_mask,
    lambda: create_classification_message(["person", "car"], [0.8, 0.2]),
    lambda: create_cluster_message([[[0.1, 0.2], [0.3, 0.4]]]),
    lambda: create_keypoints_message([[0.1, 0.2]], [0.9]),
    lambda: create_line_detection_message(
        np.array([[0.1, 0.2, 0.3, 0.4]]), np.array([0.9])
    ),
    lambda: create_map_message(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)),
    lambda: create_regression_message([0.1, 0.2]),
]


def _payload(message: dai.Buffer):
    if isinstance(message, dai.ImgDetections):
        return [
            (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            for detection in message.detections
        ]
    if isinstance(message, dai.SegmentationMask):
        return message.getCvMask().copy()
    if isinstance(message, dai.beta.Classifications):
        return list(message.classes), message.scores.copy()
    if isinstance(message, dai.beta.Clusters):
        return [
            (cluster.label, [(point.x, point.y) for point in cluster.points])
            for cluster in message.clusters
        ]
    if isinstance(message, dai.beta.Keypoints):
        return [
            (keypoint.imageCoordinates.x, keypoint.imageCoordinates.y)
            for keypoint in message.getKeypoints()
        ]
    if isinstance(message, dai.beta.Lines):
        return [
            (line.startPoint.x, line.startPoint.y, line.endPoint.x, line.endPoint.y)
            for line in message.lines
        ]
    if isinstance(message, dai.beta.Map2D):
        return message.getMap().copy()
    if isinstance(message, dai.beta.Predictions):
        return [prediction.prediction for prediction in message.predictions]
    raise TypeError(f"Unexpected message type: {type(message)}")


def _payload_equal(actual, expected) -> bool:
    if isinstance(actual, np.ndarray):
        return np.array_equal(actual, expected)
    if isinstance(actual, tuple):
        return all(_payload_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


@pytest.mark.parametrize("message_factory", MESSAGE_FACTORIES)
def test_missing_source_transformation_passes_messages_through(
    message_factory: Callable[[], dai.Buffer],
):
    message = message_factory()
    timestamp = timedelta(seconds=1)
    device_timestamp = timedelta(seconds=2)
    message.setTimestamp(timestamp)
    message.setTimestampDevice(device_timestamp)
    message.setSequenceNum(42)
    original_payload = _payload(message)
    target_transformation = (
        dai.ImgTransformation().setSourceSize(100, 100).setSize(100, 100)
    )

    remapped = remap_message(message, None, target_transformation)
    mapped_by_node = CoordinatesMapper._remap_message(
        None, message, target_transformation
    )

    assert remapped is message
    assert mapped_by_node is message
    assert message.getTransformation() is None
    assert message.getTimestamp() == timestamp
    assert message.getTimestampDevice() == device_timestamp
    assert message.getSequenceNum() == 42
    assert _payload_equal(_payload(message), original_payload)


def test_explicit_source_transformation_remains_a_fallback():
    message = create_line_detection_message(
        np.array([[0.1, 0.2, 0.3, 0.4]]), np.array([0.9])
    )
    source_transformation = (
        dai.ImgTransformation().setSourceSize(100, 100).setSize(100, 100)
    )
    target_transformation = (
        dai.ImgTransformation().setSourceSize(100, 100).setSize(200, 200)
    )

    remapped = remap_message(message, source_transformation, target_transformation)

    assert remapped is not message
    assert message.getTransformation().getSize() == (100, 100)
    assert remapped.getTransformation().getSize() == (200, 200)


def test_message_transformation_is_used_without_a_fallback():
    message = create_line_detection_message(
        np.array([[0.1, 0.2, 0.3, 0.4]]), np.array([0.9])
    )
    source_transformation = (
        dai.ImgTransformation().setSourceSize(100, 100).setSize(100, 100)
    )
    target_transformation = (
        dai.ImgTransformation().setSourceSize(100, 100).setSize(200, 200)
    )
    message.setTransformation(source_transformation)

    remapped = remap_message(message, None, target_transformation)

    assert remapped is not message
    assert remapped.getTransformation().getSize() == (200, 200)
