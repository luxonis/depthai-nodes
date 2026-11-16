import depthai as dai
import numpy as np
import pytest

import depthai_nodes.message
import depthai_nodes.node
from depthai_nodes.node import AutoSupportPlaneConfig, SupportPlaneCalibrationConfig


def test_configuration_is_exported_as_a_message():
    assert "AutoSupportPlaneCalibration" in depthai_nodes.node.__all__
    assert "AutoSupportPlaneConfig" in depthai_nodes.message.__all__
    assert AutoSupportPlaneConfig is depthai_nodes.message.AutoSupportPlaneConfig
    assert isinstance(AutoSupportPlaneConfig(), dai.Buffer)


def test_configuration_stores_built_in_numbers():
    config = AutoSupportPlaneConfig(
        check_interval_sec=np.float32(0.25),  # pyright: ignore[reportArgumentType]
        failed_checks_before_refit=np.int64(5),  # pyright: ignore[reportArgumentType]
    )

    assert config.check_interval_sec.__class__ is float
    assert config.check_interval_sec == 0.25
    assert config.failed_checks_before_refit.__class__ is int
    assert config.failed_checks_before_refit == 5


def test_default_configuration_uses_shared_assessment():
    assert AutoSupportPlaneConfig().calibration == SupportPlaneCalibrationConfig()


@pytest.mark.parametrize(
    "name,value",
    [
        ("mode", "continuous"),
        ("mode", None),
        ("check_interval_sec", True),
        ("check_interval_sec", "0.5"),
        ("failed_checks_before_refit", True),
        ("failed_checks_before_refit", 3.0),
        ("calibration", None),
        ("vertical_distance_range_mm", 700),
        ("min_on_plane_ratio", True),
        ("min_sampled_point_count", 500.0),
        ("check_distance_mm", "30"),
        ("max_gravity_angle_deg", False),
        ("calibration", {"check_distance_mm": 30.0}),
    ],
)
def test_configuration_rejects_wrong_types(name, value):
    with pytest.raises(TypeError):
        AutoSupportPlaneConfig(**{name: value})


@pytest.mark.parametrize(
    "name,value",
    [
        ("check_interval_sec", 0),
        ("check_interval_sec", -0.5),
        ("check_interval_sec", float("nan")),
        ("check_interval_sec", float("inf")),
        ("failed_checks_before_refit", 0),
        ("vertical_distance_range_mm", (900, 700)),
        ("min_on_plane_ratio", 1.1),
        ("min_sampled_point_count", 0),
        ("check_distance_mm", -1),
        ("max_gravity_angle_deg", 91),
    ],
)
def test_configuration_rejects_out_of_range_values(name, value):
    with pytest.raises(ValueError):
        AutoSupportPlaneConfig(**{name: value})
