import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from depthai_nodes.message import SupportPlane, SupportPlaneCandidate


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


FLOOR = SupportPlane(normal=(0, 0, 1), offset_mm=-1000)


def test_candidate_stores_numpy_numbers_as_builtin_types():
    candidate = SupportPlaneCandidate(
        plane=FLOOR,
        inlier_count=np.int64(120),
        inlier_ratio=np.float32(0.25),  # pyright: ignore[reportArgumentType]
    )

    assert candidate.inlier_count.__class__ is int
    assert candidate.inlier_ratio.__class__ is float


def test_candidate_accepts_a_ratio_of_one():
    assert (
        SupportPlaneCandidate(plane=FLOOR, inlier_count=1, inlier_ratio=1).inlier_ratio
        == 1.0
    )


def test_candidate_is_immutable_and_keyword_only():
    candidate = SupportPlaneCandidate(plane=FLOOR, inlier_count=1, inlier_ratio=0.5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.inlier_count = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        SupportPlaneCandidate(FLOOR, 1, 0.5)  # type: ignore[misc]


@pytest.mark.parametrize(
    "plane,count,ratio",
    [
        ((0, 0, 1), 1, 0.5),
        (None, 1, 0.5),
        (FLOOR, True, 0.5),
        (FLOOR, 1.0, 0.5),
        (FLOOR, "1", 0.5),
        (FLOOR, 1, True),
        (FLOOR, 1, "0.5"),
        (FLOOR, 1, None),
    ],
)
def test_candidate_rejects_wrong_types(plane, count, ratio):
    with pytest.raises(TypeError):
        SupportPlaneCandidate(plane=plane, inlier_count=count, inlier_ratio=ratio)


@pytest.mark.parametrize(
    "count,ratio",
    [
        (0, 0.5),
        (-1, 0.5),
        (1, 0),
        (1, 0.0),
        (1, -0.1),
        (1, 1.01),
        (1, float("nan")),
        (1, float("inf")),
    ],
)
def test_candidate_rejects_out_of_range_values(count, ratio):
    with pytest.raises(ValueError):
        SupportPlaneCandidate(plane=FLOOR, inlier_count=count, inlier_ratio=ratio)
