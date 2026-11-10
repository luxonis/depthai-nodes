import dataclasses

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import (
    CalibrationOutcome,
    SupportPlane,
    SupportPlaneCalibrationControl,
    SupportPlaneCalibrationResult,
    SupportPlaneCandidate,
    SupportPlaneVerificationResult,
)

PLANE = SupportPlane(normal=(0, 0, 1), offset_mm=-1000)
CANDIDATE = SupportPlaneCandidate(plane=PLANE, inlier_count=200, inlier_ratio=0.5)
Commands = SupportPlaneCalibrationControl.Commands


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


def test_result_carries_outcome_and_candidates():
    result = SupportPlaneCalibrationResult(CalibrationOutcome.SUCCEEDED, [CANDIDATE])

    assert isinstance(result, dai.Buffer)
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert result.candidates == (CANDIDATE,)


def test_result_has_no_info():
    result = SupportPlaneCalibrationResult(CalibrationOutcome.FAILED, [])

    assert not hasattr(result, "info")
    with pytest.raises(TypeError):
        SupportPlaneCalibrationResult(CalibrationOutcome.FAILED, [], "detail")  # type: ignore[call-arg]


def test_result_candidates_are_a_tuple_copied_at_construction():
    candidates = [CANDIDATE]
    result = SupportPlaneCalibrationResult(CalibrationOutcome.SUCCEEDED, candidates)
    candidates.append(CANDIDATE)

    assert result.candidates == (CANDIDATE,)


@pytest.mark.parametrize(
    "outcome,candidates",
    [
        (CalibrationOutcome.SUCCEEDED, []),
        (CalibrationOutcome.FAILED, [CANDIDATE]),
        (CalibrationOutcome.CANCELLED, [CANDIDATE]),
    ],
)
def test_result_has_candidates_exactly_when_calibration_succeeded(outcome, candidates):
    with pytest.raises(ValueError):
        SupportPlaneCalibrationResult(outcome, candidates)


@pytest.mark.parametrize(
    "outcome,candidates",
    [
        ("cancelled", []),
        (CalibrationOutcome.SUCCEEDED, [(0, 0, 1)]),
        (CalibrationOutcome.SUCCEEDED, PLANE),
        (CalibrationOutcome.SUCCEEDED, [PLANE]),
        (CalibrationOutcome.SUCCEEDED, CANDIDATE),
    ],
)
def test_result_rejects_wrong_types(outcome, candidates):
    with pytest.raises(TypeError):
        SupportPlaneCalibrationResult(outcome, candidates)


def test_verification_result_carries_its_fields():
    result = SupportPlaneVerificationResult(
        PLANE,
        np.float32(3.0),  # pyright: ignore[reportArgumentType]
        np.bool_(True),
        np.float64(0.75),  # pyright: ignore[reportArgumentType]
    )

    assert isinstance(result, dai.Buffer)
    assert result.plane == PLANE
    assert result.gravity_angle_deg.__class__ is float
    assert result.gravity_angle_deg == 3.0
    assert result.accepted is True
    assert result.on_plane_ratio.__class__ is float
    assert result.on_plane_ratio == 0.75


@pytest.mark.parametrize("ratio", [0, 1, 0.5])
def test_verification_result_accepts_on_plane_ratios_from_0_to_1(ratio):
    result = SupportPlaneVerificationResult(PLANE, 3.0, True, ratio)

    assert result.on_plane_ratio == ratio


def test_verification_result_on_plane_ratio_none_means_not_measured():
    assert SupportPlaneVerificationResult(PLANE, 3.0, True, None).on_plane_ratio is None


@pytest.mark.parametrize("ratio", [-0.1, 1.01, float("nan"), float("inf")])
def test_verification_result_rejects_out_of_range_on_plane_ratios(ratio):
    with pytest.raises(ValueError):
        SupportPlaneVerificationResult(PLANE, 3.0, True, ratio)


@pytest.mark.parametrize("ratio", [True, "0.5", 1j])
def test_verification_result_rejects_non_real_on_plane_ratios(ratio):
    with pytest.raises(TypeError):
        SupportPlaneVerificationResult(PLANE, 3.0, True, ratio)


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
        SupportPlaneVerificationResult(plane, angle, accepted, 0.5)


def test_verification_result_rejects_non_finite_angle():
    with pytest.raises(ValueError):
        SupportPlaneVerificationResult(PLANE, float("nan"), False, 0.5)


@pytest.mark.parametrize(
    "message,fields",
    [
        (SupportPlaneCalibrationControl.start(), ("command",)),
        (
            SupportPlaneCalibrationResult(CalibrationOutcome.SUCCEEDED, [CANDIDATE]),
            ("outcome", "candidates"),
        ),
        (
            SupportPlaneVerificationResult(PLANE, 3.0, True, 0.5),
            ("plane", "gravity_angle_deg", "accepted", "on_plane_ratio"),
        ),
    ],
)
def test_message_fields_are_read_only(message, fields):
    for field in fields:
        with pytest.raises(AttributeError):
            setattr(message, field, None)
