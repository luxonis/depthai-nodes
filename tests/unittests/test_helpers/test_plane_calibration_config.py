import dataclasses

import numpy as np
import pytest

import depthai_nodes.message
import depthai_nodes.node
from depthai_nodes.node import PlaneCalibrationConfig


def test_configuration_is_exported_from_node_not_message():
    assert "PlaneCalibrationConfig" in depthai_nodes.node.__all__
    assert "PlaneCalibrationConfig" not in depthai_nodes.message.__all__
    assert not hasattr(depthai_nodes.message, "PlaneCalibrationConfig")


def test_configuration_has_defaults_and_is_immutable():
    config = PlaneCalibrationConfig()

    assert config.distance_threshold_mm == 10.0
    assert config.ransac_sample_points == 3
    assert config.ransac_max_iterations == 300
    assert config.max_plane_passes == 10
    assert config.min_inlier_ratio == 0.05
    assert config.voxel_size_mm == 30.0
    assert config.max_gravity_angle_deg == 6.5
    assert config.merge_distance_threshold_mm == 25.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_plane_passes = 1  # type: ignore[misc]


def test_configuration_takes_keyword_arguments_only():
    with pytest.raises(TypeError):
        PlaneCalibrationConfig(10.0)  # type: ignore[misc]


def test_configuration_stores_built_in_numbers():
    config = PlaneCalibrationConfig(
        voxel_size_mm=30,
        max_plane_passes=np.int64(4),  # pyright: ignore[reportArgumentType]
        check_distance_mm=np.int64(12),  # pyright: ignore[reportArgumentType]
        min_on_plane_ratio=np.float32(0.5),  # pyright: ignore[reportArgumentType]
        max_calibration_attempts=np.int64(2),  # pyright: ignore[reportArgumentType]
    )

    assert config.voxel_size_mm.__class__ is float
    assert config.voxel_size_mm == 30.0
    assert config.max_plane_passes.__class__ is int
    assert config.max_plane_passes == 4
    assert config.check_distance_mm.__class__ is float
    assert config.check_distance_mm == 12.0
    assert config.min_on_plane_ratio.__class__ is float
    assert config.min_on_plane_ratio == 0.5
    assert config.max_calibration_attempts.__class__ is int
    assert config.max_calibration_attempts == 2


INT_FIELDS = ("ransac_sample_points", "ransac_max_iterations", "max_plane_passes")
FLOAT_FIELDS = (
    "distance_threshold_mm",
    "min_inlier_ratio",
    "voxel_size_mm",
    "max_gravity_angle_deg",
    "merge_distance_threshold_mm",
    "check_distance_mm",
    "min_on_plane_ratio",
)


@pytest.mark.parametrize(
    "name,value",
    [
        *(
            (name, value)
            for name in INT_FIELDS
            for value in (3.5, 3.0, "three", None, True)
        ),
        # The attempt cap takes None for no cap.
        *(("max_calibration_attempts", value) for value in (3.5, 3.0, "three", True)),
    ],
)
def test_configuration_rejects_wrong_int_types(name, value):
    with pytest.raises(TypeError):
        PlaneCalibrationConfig(**{name: value})


@pytest.mark.parametrize("name", FLOAT_FIELDS)
@pytest.mark.parametrize("value", ["one", None, False])
def test_configuration_rejects_wrong_float_types(name, value):
    with pytest.raises(TypeError):
        PlaneCalibrationConfig(**{name: value})


@pytest.mark.parametrize(
    "name,value",
    [
        *(
            (name, value)
            for name in (*INT_FIELDS, "max_calibration_attempts")
            for value in (0, -1)
        ),
        *((name, value) for name in FLOAT_FIELDS for value in (float("nan"), -1.0)),
        ("ransac_sample_points", 2),
        ("distance_threshold_mm", 0),
        ("voxel_size_mm", 0),
        ("merge_distance_threshold_mm", 0),
        ("check_distance_mm", 0),
        ("min_inlier_ratio", 1.5),
        ("min_on_plane_ratio", 1.5),
        ("max_gravity_angle_deg", 90),
    ],
)
def test_configuration_rejects_out_of_range_values(name, value):
    with pytest.raises(ValueError):
        PlaneCalibrationConfig(**{name: value})


def test_configuration_accepts_range_boundaries():
    config = PlaneCalibrationConfig(
        min_inlier_ratio=1,
        max_gravity_angle_deg=0,
        ransac_max_iterations=1,
        min_on_plane_ratio=1,
        max_calibration_attempts=1,
    )

    assert config.min_inlier_ratio == 1.0
    assert config.max_gravity_angle_deg == 0.0
    assert config.min_on_plane_ratio == 1.0
    assert config.max_calibration_attempts == 1
