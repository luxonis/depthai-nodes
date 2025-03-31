import inspect
import pytest
import numpy as np
from pytest import FixtureRequest

from depthai_nodes.node.utils import copy_message
from utils import create_message

ATTRS_TO_IGNORE = [
    "transformation"
]  # TODO: remove after getTransformation() is implemented


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


def test_message_copying():
    for _, creator_function in inspect.getmembers(create_message, inspect.isfunction):
        msg = creator_function()
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
