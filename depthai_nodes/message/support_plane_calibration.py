"""Support plane calibration controls and results."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

import depthai as dai
import numpy as np

from .support_plane import SupportPlane


def _check_float(name: str, value: Any) -> float:
    """Return a finite real number, ``bool`` excluded, as a built-in ``float``.

    Raises:
        TypeError: If the value is a ``bool`` or not a ``numbers.Real``.
        ValueError: If the value is not finite.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def _check_bool(name: str, value: Any) -> bool:
    """Return a ``bool`` or numpy ``bool_`` as a built-in ``bool``.

    Raises:
        TypeError: If the value is not a ``bool`` or numpy ``bool_``.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _check_instance(name: str, value: Any, expected: type) -> None:
    """Check that a value is an instance of the expected type.

    Raises:
        TypeError: If the value is not an instance of ``expected``.
    """
    if not isinstance(value, expected):
        raise TypeError(
            f"{name} must be {expected.__name__}, got {type(value).__name__}"
        )


class CalibrationOutcome(str, Enum):
    """How a calibration attempt ended.

    ``SUCCEEDED`` attempts found at least one candidate plane, ``FAILED``
    attempts found none, and ``CANCELLED`` attempts were stopped.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SupportPlaneCalibrationControl(dai.Buffer):
    """A control message for ``SupportPlaneCalibrator`` carrying one command.

    Create controls with `start`, `stop`, `reset` and `verify`. Like DepthAI's
    ``DynamicCalibrationControl``, each control carries one command.

    Args:
        command: One of the `Commands`.

    Raises:
        TypeError: If ``command`` is not one of the `Commands`.
    """

    class Commands:
        """The commands a calibration control can carry."""

        @dataclass(frozen=True, kw_only=True)
        class Start:
            """Begin an attempt unless one is running or awaits reset."""

        @dataclass(frozen=True, kw_only=True)
        class Stop:
            """Cancel the running attempt and return to idle without reset."""

        @dataclass(frozen=True, kw_only=True)
        class Reset:
            """Allow a new attempt after the previous one succeeded or failed."""

        @dataclass(frozen=True, kw_only=True)
        class Verify:
            """Check a plane against gravity without affecting any attempt.

            Attributes:
                plane: The support plane to check.

            Raises:
                TypeError: If ``plane`` is not a `SupportPlane`.
            """

            plane: SupportPlane

            def __post_init__(self) -> None:
                """Check the plane type.

                Raises:
                    TypeError: If ``plane`` is not a `SupportPlane`.
                """
                _check_instance("plane", self.plane, SupportPlane)

    Command: TypeAlias = (
        Commands.Start | Commands.Stop | Commands.Reset | Commands.Verify
    )
    """Any of the `Commands`."""

    def __init__(self, command: Command) -> None:
        """Create a control carrying ``command``.

        Args:
            command: One of the `Commands`.

        Raises:
            TypeError: If ``command`` is not one of the `Commands`.
        """
        super().__init__()
        if not isinstance(command, SupportPlaneCalibrationControl.Command):
            raise TypeError(
                f"Unknown calibration control command {type(command).__name__}"
            )
        self._command = command

    @property
    def command(self) -> Command:
        """The command this control carries."""
        return self._command

    @classmethod
    def start(cls) -> SupportPlaneCalibrationControl:
        """Create a control that begins a calibration attempt.

        Returns:
            A control carrying `Commands.Start`.
        """
        return cls(cls.Commands.Start())

    @classmethod
    def stop(cls) -> SupportPlaneCalibrationControl:
        """Create a control that cancels the running attempt.

        Returns:
            A control carrying `Commands.Stop`.
        """
        return cls(cls.Commands.Stop())

    @classmethod
    def reset(cls) -> SupportPlaneCalibrationControl:
        """Create a control that allows a new attempt after a finished one.

        Returns:
            A control carrying `Commands.Reset`.
        """
        return cls(cls.Commands.Reset())

    @classmethod
    def verify(cls, plane: SupportPlane) -> SupportPlaneCalibrationControl:
        """Create a control that checks a plane against gravity.

        Args:
            plane: The support plane to check.

        Returns:
            A control carrying `Commands.Verify`.

        Raises:
            TypeError: If ``plane`` is not a `SupportPlane`.
        """
        return cls(cls.Commands.Verify(plane=plane))


class SupportPlaneCalibrationResult(dai.Buffer):
    """The end of a calibration attempt.

    A result has candidates exactly when calibration succeeded.

    ``SupportPlaneCalibrator`` stamps each result with the timestamp, device
    timestamp and sequence number of the last point cloud its attempt fitted:
    for ``SUCCEEDED``, the cloud that yielded the candidates; for
    ``CANCELLED``, the last cloud fitted before Stop. A result whose attempt
    fitted no cloud keeps DepthAI's unset default stamps.

    Args:
        outcome: How the attempt ended.
        candidates: Accepted calibration candidates, best first.
        info: Human-readable detail about the outcome.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the result has candidates and did not succeed, or
            succeeded without candidates.
    """

    def __init__(
        self,
        outcome: CalibrationOutcome,
        candidates: list[SupportPlane] | tuple[SupportPlane, ...],
        info: str = "",
    ) -> None:
        """Create a result and check that its fields are consistent.

        Args:
            outcome: How the attempt ended.
            candidates: Accepted calibration candidates, best first.
            info: Human-readable detail about the outcome.

        Raises:
            TypeError: If ``outcome`` is not a `CalibrationOutcome`,
                ``candidates`` is not a list or tuple of `SupportPlane`, or
                ``info`` is not a string.
            ValueError: If the result has candidates and did not succeed, or
                succeeded without candidates.
        """
        super().__init__()
        _check_instance("outcome", outcome, CalibrationOutcome)
        if not isinstance(candidates, (list, tuple)):
            raise TypeError("Candidates must be a list or tuple of support planes")
        for plane in candidates:
            _check_instance("candidate", plane, SupportPlane)
        _check_instance("info", info, str)
        if bool(candidates) != (outcome is CalibrationOutcome.SUCCEEDED):
            raise ValueError("A result has candidates exactly when it succeeded")
        self._outcome = outcome
        self._candidates = tuple(candidates)
        self._info = info

    @property
    def outcome(self) -> CalibrationOutcome:
        """How the attempt ended."""
        return self._outcome

    @property
    def candidates(self) -> tuple[SupportPlane, ...]:
        """Accepted planes, best first; empty unless calibration succeeded."""
        return self._candidates

    @property
    def info(self) -> str:
        """Human-readable detail about the outcome."""
        return self._info


class SupportPlaneVerificationResult(dai.Buffer):
    """Whether a supplied plane passes calibration's gravity check.

    Args:
        plane: The checked support plane, as supplied.
        gravity_angle_deg: Tilt of the plane from horizontal, in degrees, from
            0 to 90. Either sign of the plane equation gives the same angle.
        accepted: Whether the plane passed the check.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``gravity_angle_deg`` is not finite.
    """

    def __init__(
        self,
        plane: SupportPlane,
        gravity_angle_deg: float,
        accepted: bool | np.bool_,
    ) -> None:
        """Create a verification result.

        Args:
            plane: The checked support plane, as supplied.
            gravity_angle_deg: Tilt of the plane from horizontal, in degrees.
            accepted: Whether the plane passed the check. A numpy ``bool_`` is
                stored as ``bool``.

        Raises:
            TypeError: If ``plane`` is not a `SupportPlane`,
                ``gravity_angle_deg`` is not a real number, or ``accepted`` is
                not a boolean.
            ValueError: If ``gravity_angle_deg`` is not finite.
        """
        super().__init__()
        _check_instance("plane", plane, SupportPlane)
        self._gravity_angle_deg = _check_float("gravity_angle_deg", gravity_angle_deg)
        self._accepted = _check_bool("accepted", accepted)
        self._plane = plane

    @property
    def plane(self) -> SupportPlane:
        """The checked support plane, as supplied."""
        return self._plane

    @property
    def gravity_angle_deg(self) -> float:
        """Tilt of the plane from horizontal, in degrees, from 0 to 90."""
        return self._gravity_angle_deg

    @property
    def accepted(self) -> bool:
        """Whether the plane passed the gravity check."""
        return self._accepted
