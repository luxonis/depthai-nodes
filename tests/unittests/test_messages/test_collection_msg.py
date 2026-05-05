import depthai as dai
import pytest

from depthai_nodes.message import Collection


def test_collection_infers_item_cls_from_items():
    frames = [dai.ImgFrame(), dai.ImgFrame()]

    collection = Collection(items=frames)

    assert collection.items == frames
    assert collection.item_cls is dai.ImgFrame


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
