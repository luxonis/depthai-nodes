from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import Collection
from depthai_nodes.message.creators import (
    create_classification_message,
    create_cluster_message,
    create_keypoints_message,
    create_line_detection_message,
    create_map_message,
    create_regression_message,
)
from depthai_nodes.message.utils import copy_message


def test_collection_infers_item_cls_from_items():
    frames = [dai.ImgFrame(), dai.ImgFrame()]

    collection = Collection(items=frames)

    assert collection.items == frames
    assert collection.item_cls is dai.ImgFrame


@pytest.mark.parametrize("copy_collection", [Collection.copy, copy_message])
@pytest.mark.parametrize(
    "create_message,read_payload",
    [
        (
            lambda: create_classification_message(["cat", "dog"], [0.8, 0.2]),
            lambda msg: (list(msg.classes), msg.scores.tolist()),
        ),
        (
            lambda: create_cluster_message([[[0.1, 0.2], [0.3, 0.4]]]),
            lambda msg: [
                (cluster.label, [(p.x, p.y) for p in cluster.points])
                for cluster in msg.clusters
            ],
        ),
        (
            lambda: create_keypoints_message([[0.1, 0.2]], [0.9]),
            lambda msg: [
                (p.imageCoordinates.x, p.imageCoordinates.y, p.confidence)
                for p in msg.getKeypoints()
            ],
        ),
        (
            lambda: create_line_detection_message(
                np.array([[0.1, 0.2, 0.3, 0.4]]), np.array([0.9])
            ),
            lambda msg: [
                (
                    p.startPoint.x,
                    p.startPoint.y,
                    p.endPoint.x,
                    p.endPoint.y,
                    p.confidence,
                )
                for p in msg.lines
            ],
        ),
        (
            lambda: create_map_message(np.array([[0.1, 0.2]], dtype=np.float32)),
            lambda msg: msg.getMap().tolist(),
        ),
        (
            lambda: create_regression_message([0.1, 0.2]),
            lambda msg: [p.prediction for p in msg.predictions],
        ),
    ],
    ids=["classifications", "clusters", "keypoints", "lines", "map", "predictions"],
)
def test_collection_copy_native_messages(copy_collection, create_message, read_payload):
    message = create_message()
    message.setSequenceNum(7)
    message.setTimestamp(timedelta(seconds=1))
    message.setTimestampDevice(timedelta(seconds=2))
    message.setTransformation(
        dai.ImgTransformation().setSourceSize(100, 100).setSize(100, 100)
    )
    collection = Collection([message])
    collection.setSequenceNum(42)
    collection.setTimestamp(timedelta(seconds=3))
    collection.setTimestampDevice(timedelta(seconds=4))

    copied = copy_collection(collection)

    assert copied is not collection
    assert copied.items is not collection.items
    assert copied.item_cls is collection.item_cls
    assert len(copied.items) == 1
    child = copied.items[0]
    assert child is not message
    assert read_payload(child) == read_payload(message)
    assert child.getTransformation().getSize() == (100, 100)
    for original, duplicate in [(collection, copied), (message, child)]:
        assert duplicate.getSequenceNum() == original.getSequenceNum()
        assert duplicate.getTimestamp() == original.getTimestamp()
        assert duplicate.getTimestampDevice() == original.getTimestampDevice()


def test_collection_copy_native_map_is_independent():
    message = create_map_message(np.array([[1, 2], [3, 4]], dtype=np.float32))
    collection = Collection([message])

    copied = copy_message(collection)
    copied.items[0].getMap()[:] = 9
    copied.items[0].setSequenceNum(99)
    copied.append(create_map_message(np.zeros((2, 2), dtype=np.float32)))

    np.testing.assert_array_equal(message.getMap(), [[1, 2], [3, 4]])
    assert message.getSequenceNum() != 99
    assert len(collection.items) == 1


def test_empty_collection_copy_preserves_metadata():
    collection = Collection([])
    collection.setSequenceNum(42)
    collection.setTimestamp(timedelta(seconds=3))
    collection.setTimestampDevice(timedelta(seconds=4))

    copied = copy_message(collection)

    assert copied is not collection
    assert copied.items == []
    assert copied.item_cls is None
    assert copied.getSequenceNum() == 42
    assert copied.getTimestamp() == collection.getTimestamp()
    assert copied.getTimestampDevice() == collection.getTimestampDevice()


def test_collection_rejects_mixed_item_types():
    with pytest.raises(TypeError):
        Collection(items=[dai.ImgFrame(), dai.Buffer()])


def test_collection_empty_list_infers_on_first_append():
    collection = Collection(items=[])
    frame = dai.ImgFrame()

    assert collection.item_cls is None

    collection.append(frame)

    assert collection.item_cls is dai.ImgFrame
    assert collection.items == [frame]


def test_collection_empty_list_infers_on_first_assignment():
    collection = Collection(items=[])
    frames = [dai.ImgFrame()]

    collection.items = frames

    assert collection.item_cls is dai.ImgFrame
    assert collection.items == frames
