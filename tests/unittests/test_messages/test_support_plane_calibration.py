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


def test_result_carries_outcome_candidates_and_info():
    result = SupportPlaneCalibrationResult(
        CalibrationOutcome.SUCCEEDED, [CANDIDATE], "found"
    )

    assert isinstance(result, dai.Buffer)
    assert result.outcome is CalibrationOutcome.SUCCEEDED
    assert result.candidates == (CANDIDATE,)
    assert result.info == "found"


def test_result_info_defaults_to_empty():
    assert SupportPlaneCalibrationResult(CalibrationOutcome.FAILED, []).info == ""


def test_result_candidates_are_a_tuple_copied_at_construction():
    candidates = [CANDIDATE]
    result = SupportPlaneCalibrationResult(
        CalibrationOutcome.SUCCEEDED, candidates, "found"
    )
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
    "outcome,candidates,info",
    [
        ("cancelled", [], ""),
        (CalibrationOutcome.SUCCEEDED, [(0, 0, 1)], ""),
        (CalibrationOutcome.SUCCEEDED, PLANE, ""),
        (CalibrationOutcome.SUCCEEDED, [PLANE], ""),
        (CalibrationOutcome.SUCCEEDED, CANDIDATE, ""),
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
                CalibrationOutcome.SUCCEEDED, [CANDIDATE], "found"
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
