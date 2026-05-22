from datetime import timedelta

import depthai as dai

from depthai_nodes.message import GatheredData


def create_reference() -> dai.Buffer:
    reference = dai.Buffer()
    reference.setSequenceNum(42)
    reference.setTimestamp(timedelta(seconds=5))
    reference.setTimestampDevice(timedelta(seconds=7))
    return reference


def test_gathered_data_initialization_with_items():
    reference = create_reference()
    items = [dai.ImgFrame(), dai.ImgFrame()]

    gathered_data = GatheredData(reference_data=reference, items=items)

    assert gathered_data.reference_data is reference
    assert gathered_data.items == items
    assert gathered_data.item_cls is dai.ImgFrame


def test_gathered_data_initialization_with_empty_items():
    reference = create_reference()

    gathered_data = GatheredData(reference_data=reference, items=[])

    assert gathered_data.reference_data is reference
    assert gathered_data.items == []
    assert gathered_data.item_cls is None


def test_gathered_data_copies_metadata_from_reference():
    reference = create_reference()
    item = dai.Buffer()

    gathered_data = GatheredData(reference_data=reference, items=[item])

    assert gathered_data.getSequenceNum() == reference.getSequenceNum()
    assert gathered_data.getTimestamp() == reference.getTimestamp()
    assert gathered_data.getTimestampDevice() == reference.getTimestampDevice()
