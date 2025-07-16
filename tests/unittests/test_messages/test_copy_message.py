import inspect
from typing import Callable, List, Tuple

import numpy as np
import pytest

from depthai_nodes.message.utils import copy_message
from tests.utils.messages import creators as message_creators

ATTRS_TO_IGNORE = [
    "Type",  # dai.ImgFrame attribute
]


def equal_attributes(obj1, obj2):
    if type(obj1) is not type(obj2):
        return False
    if np.isscalar(obj1):
        # int, float, str, bool, bytes, np.int32, np.float64, np.bool_, etc.
        return obj1 == obj2
    elif isinstance(obj1, np.ndarray):
        return np.array_equal(obj1, obj2)
    elif isinstance(obj1, (List, Tuple)):
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
    "message_creator",
    message_creators.__all__,
)
def test_message_copying(message_creator: Tuple[str, Callable]):
    creator_function = getattr(message_creators, message_creator)

    msg = creator_function()
    try:
        msg_copy = copy_message(msg)
        assert isinstance(msg_copy, type(msg))
        assert equal_attributes(msg, msg_copy)
        if hasattr(msg, "getSequenceNum"):
            assert msg.getSequenceNum() == msg_copy.getSequenceNum()
        if hasattr(msg, "getTimestamp"):
            assert msg.getTimestamp() == msg_copy.getTimestamp()
        if hasattr(msg, "getTimestampDevice"):
            assert msg.getTimestampDevice() == msg_copy.getTimestampDevice()
        if hasattr(msg, "getTransformation"):
            assert equal_attributes(
                msg_copy.getTransformation(), msg.getTransformation()
            )  # comparisson of dai.ImgTransformation objects

    except TypeError:  # copying not implemented for all messages
        pass
