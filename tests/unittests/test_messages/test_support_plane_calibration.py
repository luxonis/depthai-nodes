import dataclasses

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import (
    CalibrationOutcome,
    PlaneCalibrationConfig,
    SupportPlane,
    SupportPlaneCalibrationControl,
    SupportPlaneCalibrationResult,
    SupportPlaneVerificationResult,
)

PLANE = SupportPlane(normal=(0, 0, 1), offset_mm=-1000)
Commands = SupportPlaneCalibrationControl.Commands


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
    config = PlaneCalibrationConfig(voxel_size_mm=30, max_plane_passes=np.int64(4))  # pyright: ignore[reportArgumentType]

    assert config.voxel_size_mm.__class__ is float
    assert config.voxel_size_mm == 30.0
    assert config.max_plane_passes.__class__ is int
    assert config.max_plane_passes == 4


INT_FIELDS = ("ransac_sample_points", "ransac_max_iterations", "max_plane_passes")
FLOAT_FIELDS = (
    "distance_threshold_mm",
    "min_inlier_ratio",
    "voxel_size_mm",
    "max_gravity_angle_deg",
    "merge_distance_threshold_mm",
)


@pytest.mark.parametrize("name", INT_FIELDS)
@pytest.mark.parametrize("value", [3.5, 3.0, "three", None, True])
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
        *((name, value) for name in INT_FIELDS for value in (0, -1)),
        *((name, value) for name in FLOAT_FIELDS for value in (float("nan"), -1.0)),
        ("ransac_sample_points", 2),
        ("distance_threshold_mm", 0),
        ("voxel_size_mm", 0),
        ("merge_distance_threshold_mm", 0),
        ("min_inlier_ratio", 1.5),
        ("max_gravity_angle_deg", 90),
    ],
)
def test_configuration_rejects_out_of_range_values(name, value):
    with pytest.raises(ValueError):
        PlaneCalibrationConfig(**{name: value})


def test_configuration_accepts_range_boundaries():
    config = PlaneCalibrationConfig(
        min_inlier_ratio=1, max_gravity_angle_deg=0, ransac_max_iterations=1
    )

    assert config.min_inlier_ratio == 1.0
    assert config.max_gravity_angle_deg == 0.0


def test_controls_are_buffers_carrying_one_command():
    assert isinstance(SupportPlaneCalibrationControl.start(), dai.Buffer)
    assert isinstance(SupportPlaneCalibrationControl.start().command, Commands.Start)
    assert isinstance(SupportPlaneCalibrationControl.stop().command, Commands.Stop)
    assert isinstance(SupportPlaneCalibrationControl.reset().command, Commands.Reset)
    command = SupportPlaneCalibrationControl.verify(PLANE).command
    assert isinstance(command, Commands.Verify)
    assert command.plane == PLANE


def test_commands_are_immutable_and_keyword_only():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SupportPlaneCalibrationControl.verify(PLANE).command.plane = PLANE  # type: ignore[misc]
    with pytest.raises(TypeError):
        Commands.Verify(PLANE)  # type: ignore[misc]


def test_verify_command_requires_a_support_plane():
    with pytest.raises(TypeError):
        Commands.Verify(plane=(0, 0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SupportPlaneCalibrationControl.verify(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("command", ["start", None, object()])
def test_controls_reject_unknown_commands(command):
    with pytest.raises(TypeError):
        SupportPlaneCalibrationControl(command)


def test_outcomes_are_their_string_values():
    assert CalibrationOutcome.SUCCEEDED == "succeeded"
    assert CalibrationOutcome.FAILED == "failed"
    assert CalibrationOutcome.CANCELLED == "cancelled"


def test_result_carries_outcome_candidates_and_info():
    result = SupportPlaneCalibrationResult(
        CalibrationOutcome.SUCCEEDED, [PLANE], "found"
    )

    assert isinstance(result, dai.Buffer)
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert result.candidates == (PLANE,)
    assert result.info == "found"


def test_result_info_defaults_to_empty():
    assert SupportPlaneCalibrationResult(CalibrationOutcome.FAILED, []).info == ""


def test_result_candidates_are_a_tuple_copied_at_construction():
    candidates = [PLANE]
    result = SupportPlaneCalibrationResult(
        CalibrationOutcome.SUCCEEDED, candidates, "found"
    )
    candidates.append(PLANE)

    assert result.candidates == (PLANE,)


@pytest.mark.parametrize(
    "outcome,candidates",
    [
        (CalibrationOutcome.SUCCEEDED, []),
        (CalibrationOutcome.FAILED, [PLANE]),
        (CalibrationOutcome.CANCELLED, [PLANE]),
    ],
)
def test_result_has_candidates_exactly_when_calibration_succeeded(outcome, candidates):
    with pytest.raises(ValueError):
        SupportPlaneCalibrationResult(outcome, candidates)


@pytest.mark.parametrize(
    "outcome,candidates,info",
    [
        ("cancelled", [], ""),
        (CalibrationOutcome.SUCCEEDED, [(0, 0, 1)], ""),
        (CalibrationOutcome.SUCCEEDED, PLANE, ""),
        (CalibrationOutcome.FAILED, [], None),
    ],
)
def test_result_rejects_wrong_types(outcome, candidates, info):
    with pytest.raises(TypeError):
        SupportPlaneCalibrationResult(outcome, candidates, info)


def test_verification_result_carries_its_fields():
    result = SupportPlaneVerificationResult(PLANE, np.float32(3.0), np.bool_(True))  # pyright: ignore[reportArgumentType]

    assert isinstance(result, dai.Buffer)
    assert result.plane == PLANE
    assert result.gravity_angle_deg.__class__ is float
    assert result.gravity_angle_deg == 3.0
    assert result.accepted is True


@pytest.mark.parametrize(
    "plane,angle,accepted",
    [
        ((0, 0, 1), 3.0, True),
        (PLANE, "3", True),
        (PLANE, True, True),
        (PLANE, 3.0, 1),
        (PLANE, 3.0, "yes"),
    ],
)
def test_verification_result_rejects_wrong_types(plane, angle, accepted):
    with pytest.raises(TypeError):
        SupportPlaneVerificationResult(plane, angle, accepted)


def test_verification_result_rejects_non_finite_angle():
    with pytest.raises(ValueError):
        SupportPlaneVerificationResult(PLANE, float("nan"), False)


@pytest.mark.parametrize(
    "message,fields",
    [
        (SupportPlaneCalibrationControl.start(), ("command",)),
        (
            SupportPlaneCalibrationResult(
                CalibrationOutcome.SUCCEEDED, [PLANE], "found"
            ),
            ("outcome", "candidates", "info"),
        ),
        (
            SupportPlaneVerificationResult(PLANE, 3.0, True),
            ("plane", "gravity_angle_deg", "accepted"),
        ),
    ],
)
def test_message_fields_are_read_only(message, fields):
    for field in fields:
        with pytest.raises(AttributeError):
            setattr(message, field, None)
