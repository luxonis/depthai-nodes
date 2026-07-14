import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message.creators import create_segmentation_message

MASK = np.random.randint(0, 256, (480, 640), dtype=np.uint8)


def test_valid_input():
    message = create_segmentation_message(MASK)
    mask = message.getCvMask()

    assert isinstance(message, dai.SegmentationMask)
    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8


def test_invalid_type():
    with pytest.raises(ValueError):
        create_segmentation_message(MASK.tolist())


def test_invalid_shape():
    mask = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        create_segmentation_message(mask)


def test_invalid_dtype():
    with pytest.raises(ValueError):
        create_segmentation_message(MASK.astype(np.int16))


def test_empty_mask():
    mask = np.empty((0, 0), dtype=np.uint8)
    message = create_segmentation_message(mask)
    mask = message.getCvMask()

    assert isinstance(message, dai.SegmentationMask)
    assert mask is None
