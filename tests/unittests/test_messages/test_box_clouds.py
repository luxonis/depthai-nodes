import dataclasses

import numpy as np
import pytest

from depthai_nodes.message import BoxCloud, BoxClouds, BoxViewDetection
from tests.utils.messages.creators.box import create_rotated_rect


def detection_fields(**overrides) -> dict:
    return {
        "view": 1,
        "track_id": 7,
        "bounding_box": create_rotated_rect(),
        "confidence": 0.8,
        **overrides,
    }


def cloud_fields(**overrides) -> dict:
    return {
        "box_id": 3,
        "points": np.zeros((5, 3), dtype=np.float32),
        "depth_coverage": 0.9,
        "detections": (BoxViewDetection(**detection_fields()),),
        **overrides,
    }


def test_detection_stores_numbers_as_built_ins_and_is_frozen():
    detection = BoxViewDetection(
        **detection_fields(view=np.int64(1), confidence=np.float32(0.5))
    )

    assert detection.view.__class__ is int
    assert detection.confidence.__class__ is float
    with pytest.raises(dataclasses.FrozenInstanceError):
        detection.view = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        BoxViewDetection(1, 7, create_rotated_rect(), 0.8)  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("view", True, TypeError),
        ("view", 1.0, TypeError),
        ("view", -1, ValueError),
        ("track_id", 7.5, TypeError),
        ("bounding_box", (0.5, 0.5), TypeError),
        ("confidence", "0.8", TypeError),
        ("confidence", float("nan"), ValueError),
    ],
)
def test_detection_rejects_bad_fields(field, value, error):
    with pytest.raises(error):
        BoxViewDetection(**detection_fields(**{field: value}))


def test_cloud_stores_numbers_as_built_ins_and_is_frozen():
    cloud = BoxCloud(**cloud_fields(box_id=np.int64(3), depth_coverage=np.float32(1)))

    assert cloud.box_id.__class__ is int
    assert cloud.depth_coverage.__class__ is float
    with pytest.raises(dataclasses.FrozenInstanceError):
        cloud.box_id = 4  # type: ignore[misc]
    with pytest.raises(TypeError):
        BoxCloud(3, np.zeros((0, 3), np.float32), 0.5, ())  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("box_id", 3.0, TypeError),
        ("points", [[0.0, 0.0, 0.0]], TypeError),
        ("points", np.zeros((5, 2), dtype=np.float32), ValueError),
        ("points", np.zeros(3, dtype=np.float32), ValueError),
        ("points", np.zeros((5, 3), dtype=np.float64), ValueError),
        ("depth_coverage", "0.9", TypeError),
        ("depth_coverage", float("nan"), ValueError),
        ("depth_coverage", -0.1, ValueError),
        ("depth_coverage", 1.1, ValueError),
        ("detections", [BoxViewDetection(**detection_fields())], TypeError),
        ("detections", ("detection",), TypeError),
        ("detections", (), ValueError),
    ],
)
def test_cloud_rejects_bad_fields(field, value, error):
    with pytest.raises(error):
        BoxCloud(**cloud_fields(**{field: value}))


def test_clouds_accept_only_box_clouds_from_construction():
    detection = BoxViewDetection(**detection_fields())
    with pytest.raises(TypeError):
        BoxClouds([detection])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        BoxClouds([]).append(detection)  # type: ignore[arg-type]
