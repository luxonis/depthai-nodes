import dataclasses

import numpy as np
import pytest

import depthai_nodes.message
import depthai_nodes.node
from depthai_nodes.node import AutoSupportPlaneConfig


def test_configuration_is_exported_from_node_not_message():
    assert "AutoSupportPlaneConfig" in depthai_nodes.node.__all__
    assert "AutoSupportPlaneConfig" not in depthai_nodes.message.__all__


def test_configuration_is_frozen_and_takes_keyword_arguments_only():
    config = AutoSupportPlaneConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.check_interval_sec = 1.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        AutoSupportPlaneConfig(AutoSupportPlaneConfig.Mode.CONTINUOUS)  # type: ignore[misc]


def test_configuration_stores_built_in_numbers():
    config = AutoSupportPlaneConfig(
        check_interval_sec=np.float32(0.25),  # pyright: ignore[reportArgumentType]
        failed_checks_before_refit=np.int64(5),  # pyright: ignore[reportArgumentType]
    )

    assert config.check_interval_sec.__class__ is float
    assert config.check_interval_sec == 0.25
    assert config.failed_checks_before_refit.__class__ is int
    assert config.failed_checks_before_refit == 5


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
    ],
)
def test_configuration_rejects_out_of_range_values(name, value):
    with pytest.raises(ValueError):
        AutoSupportPlaneConfig(**{name: value})
