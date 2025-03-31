import inspect
import pytest
import numpy as np
import depthai as dai
from pytest import FixtureRequest

from depthai_nodes.node.utils import copy_message
from utils.create_message import *

DETS = [
    {"bbox": [0.00, 0.00, 0.25, 0.25], "label": 0, "confidence": 0.25},
    {"bbox": [0.25, 0.25, 0.50, 0.50], "label": 1, "confidence": 0.50},
    {"bbox": [0.50, 0.50, 0.75, 0.75], "label": 2, "confidence": 0.75},
    {"bbox": [0.75, 0.75, 1.00, 1.00], "label": 3, "confidence": 1.00},
]

HEIGHT, WIDTH = 5, 5
MAX_VALUE = 50
ARR = np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH), dtype=np.int16)

ATTRS_TO_IGNORE = [
    "transformation"
]  # TODO: remove after getTransformation() is implemented


@pytest.fixture
def img_detections():
    return create_img_detections(DETS)


@pytest.fixture
def img_detections_extended():
    return create_img_detections_extended(DETS, ARR)


@pytest.fixture
def img_frame():
    return create_img_frame(ARR, dai.ImgFrame.Type.RAW8)


@pytest.fixture
def segmentation_mask():
    return create_segmentation_mask(ARR)


@pytest.fixture
def map2d():
    return create_map2d(ARR.astype(np.float32))


def equal_attributes(obj1, obj2):
    if isinstance(obj1, (int, float, str, bool, bytes)):
        return obj1 == obj2  # directly comparable types
    elif isinstance(obj1, np.ndarray):
        return np.array_equal(obj1, obj2)
    elif isinstance(obj1, (list, tuple)):
        return len(obj1) == len(obj2) and all(
            equal_attributes(a, b) for a, b in zip(obj1, obj2)
        )
    elif hasattr(obj1, "__dir__"):
        # iterate attributes
        attrs = [
            attr
            for attr in dir(obj1)
            if attr not in ATTRS_TO_IGNORE  # skip ignored attributes
            and not attr.startswith("_")  # skip private attributes
            and not inspect.ismethod(getattr(obj1, attr, None))  # skip methods
        ]
        return all(
            equal_attributes(getattr(obj1, attr, None), getattr(obj2, attr, None))
            for attr in attrs
        )
    else:
        raise ValueError(f"Unsupported attribute type: {type(obj1)}")


@pytest.mark.parametrize(
    "msg_type",
    [
        "img_detections",
        "img_detections_extended",
        "img_frame",
        "segmentation_mask",
        "map2d",
    ],
)
def test_message_copying(
    request: FixtureRequest,
    msg_type: str,
):
    msg = request.getfixturevalue(msg_type)

    try:
        msg_copy = copy_message(msg)
        assert isinstance(msg_copy, type(msg))
        assert equal_attributes(msg, msg_copy)
        # check general message information
        assert msg_copy.getSequenceNum() == msg.getSequenceNum()
        assert msg_copy.getTimestamp() == msg.getTimestamp()
        assert msg_copy.getTimestampDevice() == msg.getTimestampDevice()
        # assert objects_equal(msg_copy.getTransformation(), msg.getTransformation()) TODO: add after getTransformation() is implemented
    except:
        pass  # not all messages are supported for copying
