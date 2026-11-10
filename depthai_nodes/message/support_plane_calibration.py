"""Support plane calibration controls and results."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

import depthai as dai
import numpy as np

from .support_plane import SupportPlane, SupportPlaneCandidate


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
    attempts hit the attempt cap (``max_calibration_attempts`` in the
    calibrator's config) without one, and ``CANCELLED`` attempts were stopped.
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

    A result has candidates exactly when calibration succeeded. Candidates are
    ordered best first: most inliers first, ties going to the smallest absolute
    ``plane.offset_mm`` (nearest the camera). Each candidate's inlier count and
    ratio are measured on the preprocessed point cloud; see
    `SupportPlaneCandidate`.

    ``SupportPlaneCalibrator`` stamps each result with the timestamp, device
    timestamp and sequence number of the last point cloud its attempt fitted:
    for ``SUCCEEDED``, the cloud that yielded the candidates; for
    ``CANCELLED``, the last cloud fitted before Stop. A result whose attempt
    fitted no cloud keeps DepthAI's unset default stamps.

    Args:
        outcome: How the attempt ended.
        candidates: Support plane candidates, best first.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the result has candidates and did not succeed, or
            succeeded without candidates.
    """

    def __init__(
        self,
        outcome: CalibrationOutcome,
        candidates: list[SupportPlaneCandidate] | tuple[SupportPlaneCandidate, ...],
    ) -> None:
        """Create a result and check that its fields are consistent.

        Args:
            outcome: How the attempt ended.
            candidates: Support plane candidates, best first.

        Raises:
            TypeError: If ``outcome`` is not a `CalibrationOutcome`, or
                ``candidates`` is not a list or tuple of
                `SupportPlaneCandidate`.
            ValueError: If the result has candidates and did not succeed, or
                succeeded without candidates.
        """
        super().__init__()
        _check_instance("outcome", outcome, CalibrationOutcome)
        if not isinstance(candidates, (list, tuple)):
            raise TypeError(
                "Candidates must be a list or tuple of support plane candidates"
            )
        for candidate in candidates:
            _check_instance("candidate", candidate, SupportPlaneCandidate)
        if bool(candidates) != (outcome is CalibrationOutcome.SUCCEEDED):
            raise ValueError("A result has candidates exactly when it succeeded")
        self._outcome = outcome
        self._candidates = tuple(candidates)

    @property
    def outcome(self) -> CalibrationOutcome:
        """How the attempt ended."""
        return self._outcome

    @property
    def candidates(self) -> tuple[SupportPlaneCandidate, ...]:
        """Support plane candidates, best first; empty unless calibration succeeded.

        Ordered by most inliers, then by smallest absolute ``plane.offset_mm``.
        """
        return self._candidates


class SupportPlaneVerificationResult(dai.Buffer):
    """How a supplied plane fares against gravity and the current point cloud.

    The level test (``gravity_angle_deg`` and ``accepted``) is calibration's
    gravity check. The on-plane ratio is the share of the checked point
    cloud's sampled valid points within ``check_distance_mm`` of the plane,
    whichever sign its equation has. It is not a candidate's inlier ratio: it
    is measured on a sparse sample of the raw organized cloud, not on the
    preprocessed cloud RANSAC fits.

    Args:
        plane: The checked support plane, as supplied.
        gravity_angle_deg: Tilt of the plane from horizontal, in degrees, from
            0 to 90. Either sign of the plane equation gives the same angle.
        accepted: Whether the plane passed the gravity check.
        on_plane_ratio: The on-plane ratio from 0 to 1, or ``None`` when it
            was not measured because too few valid points were sampled.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``gravity_angle_deg`` is not finite, or
            ``on_plane_ratio`` is not finite or outside 0 to 1.
    """

    def __init__(
        self,
        plane: SupportPlane,
        gravity_angle_deg: float,
        accepted: bool | np.bool_,
        on_plane_ratio: float | None,
    ) -> None:
        """Create a verification result.

        Args:
            plane: The checked support plane, as supplied.
            gravity_angle_deg: Tilt of the plane from horizontal, in degrees.
            accepted: Whether the plane passed the gravity check. A numpy
                ``bool_`` is stored as ``bool``.
            on_plane_ratio: The on-plane ratio from 0 to 1, or ``None`` when
                not measured.

        Raises:
            TypeError: If ``plane`` is not a `SupportPlane`,
                ``gravity_angle_deg`` is not a real number, ``accepted`` is
                not a boolean, or ``on_plane_ratio`` is neither ``None`` nor a
                real number.
            ValueError: If ``gravity_angle_deg`` is not finite, or
                ``on_plane_ratio`` is not finite or outside 0 to 1.
        """
        super().__init__()
        _check_instance("plane", plane, SupportPlane)
        self._gravity_angle_deg = _check_float("gravity_angle_deg", gravity_angle_deg)
        self._accepted = _check_bool("accepted", accepted)
        if on_plane_ratio is not None:
            on_plane_ratio = _check_float("on_plane_ratio", on_plane_ratio)
            if not 0 <= on_plane_ratio <= 1:
                raise ValueError(
                    f"on_plane_ratio must be from 0 to 1, got {on_plane_ratio}"
                )
        self._on_plane_ratio = on_plane_ratio
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

    @property
    def on_plane_ratio(self) -> float | None:
        """Share of sampled valid points on the plane, or ``None`` if not measured.

        A point is on the plane within the calibrator's ``check_distance_mm``.
        ``None`` means too few valid points were sampled to measure. This is
        not a candidate's inlier ratio.
        """
        return self._on_plane_ratio
