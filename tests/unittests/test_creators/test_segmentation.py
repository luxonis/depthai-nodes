import numpy as np
import pytest

from depthai_nodes import SegmentationMask
from depthai_nodes.message.creators import create_segmentation_message

MASK = np.random.randint(0, 256, (480, 640), dtype=np.int16)


def test_valid_input():
    message = create_segmentation_message(MASK)

    assert isinstance(message, SegmentationMask)
    assert message.mask.shape == (480, 640)
    assert message.mask.dtype == np.int16


def test_invalid_type():
    with pytest.raises(ValueError):
        create_segmentation_message(MASK.tolist())


def test_invalid_shape():
    mask = np.random.randint(0, 256, (480, 640, 3), dtype=np.int16)
    with pytest.raises(ValueError):
        create_segmentation_message(mask)


def test_invalid_dtype():
    with pytest.raises(ValueError):
        create_segmentation_message(MASK.astype(np.uint8))


def test_empty_mask():
    mask = np.empty((0, 0), dtype=np.int16)
    message = create_segmentation_message(mask)

    assert isinstance(message, SegmentationMask)
    assert message.mask.shape == (0, 0)
    assert message.mask.dtype == np.int16
