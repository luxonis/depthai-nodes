import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from depthai_nodes.message import SupportPlane


def test_support_plane_normalizes_normal_and_offset_together():
    plane = SupportPlane(normal=(0, 0, 2), offset_mm=-1000)

    assert plane.normal == (0.0, 0.0, 1.0)
    assert plane.offset_mm == -500.0
    assert all(value.__class__ is float for value in (*plane.normal, plane.offset_mm))


def test_support_plane_accepts_a_list_normal_with_opposite_sign():
    plane = SupportPlane(normal=[0, 0, -2], offset_mm=1000)  # pyright: ignore[reportArgumentType]

    assert plane.normal == (0.0, 0.0, -1.0)
    assert plane.offset_mm == 500.0


def test_support_plane_is_immutable():
    plane = SupportPlane(normal=(0, 0, 1), offset_mm=-500)

    with pytest.raises(dataclasses.FrozenInstanceError):
        plane.offset_mm = 0  # type: ignore[misc]


def test_support_plane_takes_keyword_arguments_only():
    with pytest.raises(TypeError):
        SupportPlane((0, 0, 1), -500)  # type: ignore[misc]


def test_support_plane_accepts_numpy_scalars():
    plane = SupportPlane(
        normal=(np.float32(0), np.int64(0), np.float64(2)),  # pyright: ignore[reportArgumentType]
        offset_mm=np.int32(-4),  # pyright: ignore[reportArgumentType]
    )

    assert plane.normal == (0.0, 0.0, 1.0)
    assert plane.offset_mm == -2.0
    assert all(value.__class__ is float for value in (*plane.normal, plane.offset_mm))


@pytest.mark.parametrize(
    "normal,offset",
    [
        ((0, 0, 0), 1),
        ((0, 1), 1),
        ((0, 0, 1, 0), 1),
        ((0, float("nan"), 1), 1),
        ((0, 0, 1), float("inf")),
        ((1e-300, 0, 0), 1e300),
    ],
)
def test_support_plane_rejects_invalid_geometry(normal, offset):
    with pytest.raises(ValueError):
        SupportPlane(normal=normal, offset_mm=offset)


@pytest.mark.parametrize(
    "normal,offset",
    [
        ((0, 0, "up"), 1),
        ((0, 0, 1), None),
        ((0, 0, 1), "1"),
        ((0, 0, True), 1),
        ((0, 0, 1), False),
        ("001", 1),
        (None, 1),
    ],
)
def test_support_plane_rejects_non_numeric_geometry(normal, offset):
    with pytest.raises(TypeError):
        SupportPlane(normal=normal, offset_mm=offset)


def test_support_plane_json_round_trips_only_normalized_geometry():
    plane = SupportPlane(normal=(0, 0, 2), offset_mm=-1000)

    assert plane.toJson() == {"normal": [0.0, 0.0, 1.0], "offset_mm": -500.0}
    assert SupportPlane.fromJson(plane.toJson()) == plane


def test_support_plane_file_round_trip(tmp_path: Path):
    plane = SupportPlane(normal=(0, 0, 2), offset_mm=-1000)
    path = tmp_path / "plane.json"

    plane.toJsonFile(path)

    assert SupportPlane.fromJsonFile(path) == plane
    assert SupportPlane.fromJsonFile(str(path)) == plane


def test_support_plane_file_errors_propagate(tmp_path: Path):
    plane = SupportPlane(normal=(0, 0, 1), offset_mm=-500)
    path = tmp_path / "plane.json"
    path.write_text("broken json")

    with pytest.raises(json.JSONDecodeError):
        SupportPlane.fromJsonFile(path)
    with pytest.raises(OSError):
        plane.toJsonFile(tmp_path / "missing" / "plane.json")
    with pytest.raises(OSError):
        SupportPlane.fromJsonFile(tmp_path / "absent.json")


@pytest.mark.parametrize(
    "data",
    [
        {"plane_eq": [0, 0, 1, -500]},
        {"normal": [0, 0, 1]},
        {"normal": [0, 0], "offset_mm": -500},
        {"normal": [0, 0, 1], "offset_mm": -500, "version": 1},
    ],
)
def test_support_plane_rejects_incompatible_json(data):
    with pytest.raises(ValueError):
        SupportPlane.fromJson(data)


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {"normal": [0, 0, "1"], "offset_mm": -500},
        {"normal": [0, 0, 1], "offset_mm": None},
        {"normal": [0, 0, 1], "offset_mm": False},
        {"normal": "001", "offset_mm": -500},
    ],
)
def test_support_plane_rejects_json_of_the_wrong_type(data):
    with pytest.raises(TypeError):
        SupportPlane.fromJson(data)
