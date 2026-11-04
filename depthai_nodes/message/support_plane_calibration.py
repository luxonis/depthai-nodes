"""Support plane calibration settings, controls and results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

import depthai as dai
import numpy as np

from .support_plane import SupportPlane
from .utils.value_checks import check_float, check_instance, check_int


@dataclass(frozen=True, kw_only=True)
class PlaneCalibrationConfig:
    """Plane-fitting settings for ``SupportPlaneCalibrator``, fixed at build.

    Attributes:
        distance_threshold_mm: Largest distance of a RANSAC inlier from the
            plane, in millimeters. Must be positive.
        ransac_sample_points: Points sampled per RANSAC iteration. At least 3.
        ransac_max_iterations: RANSAC iterations per plane. At least 1.
        max_plane_passes: Planes extracted from one point cloud. At least 1.
        min_inlier_ratio: Smallest share of the cloud a plane must hold, from 0
            to 1.
        voxel_size_mm: Downsampling voxel size in millimeters. Must be positive.
        max_gravity_angle_deg: Largest angle between a plane normal and the
            direction opposing gravity, from 0 up to but excluding 90.
        merge_distance_threshold_mm: Largest offset difference, in millimeters,
            at which two parallel candidates count as one plane. Must be
            positive.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is not finite or out of range.
    """

    distance_threshold_mm: float = 10.0
    ransac_sample_points: int = 3
    ransac_max_iterations: int = 300
    max_plane_passes: int = 10
    min_inlier_ratio: float = 0.05
    voxel_size_mm: float = 30.0
    max_gravity_angle_deg: float = 6.5
    merge_distance_threshold_mm: float = 25.0

    def __post_init__(self) -> None:
        """Validate the fields and store numbers as built-in types.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is not finite or out of range.
        """
        minimum_ints = {
            "ransac_sample_points": 3,
            "ransac_max_iterations": 1,
            "max_plane_passes": 1,
        }
        for name, minimum in minimum_ints.items():
            value = check_int(name, getattr(self, name))
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}, got {value}")
            object.__setattr__(self, name, value)
        for name in (
            "distance_threshold_mm",
            "voxel_size_mm",
            "merge_distance_threshold_mm",
        ):
            value = check_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)
        ratio = check_float("min_inlier_ratio", self.min_inlier_ratio)
        if not 0 <= ratio <= 1:
            raise ValueError(f"min_inlier_ratio must be from 0 to 1, got {ratio}")
        object.__setattr__(self, "min_inlier_ratio", ratio)
        angle = check_float("max_gravity_angle_deg", self.max_gravity_angle_deg)
        if not 0 <= angle < 90:
            raise ValueError(
                f"max_gravity_angle_deg must be from 0 up to 90, got {angle}"
            )
        object.__setattr__(self, "max_gravity_angle_deg", angle)


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
                check_instance("plane", self.plane, SupportPlane)

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
        check_instance("outcome", outcome, CalibrationOutcome)
        if not isinstance(candidates, (list, tuple)):
            raise TypeError("Candidates must be a list or tuple of support planes")
        for plane in candidates:
            check_instance("candidate", plane, SupportPlane)
        check_instance("info", info, str)
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
        plane: The checked support plane.
        gravity_angle_deg: Angle between the plane normal and the direction
            opposing gravity, in degrees.
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
            plane: The checked support plane.
            gravity_angle_deg: Angle between the plane normal and the direction
                opposing gravity, in degrees.
            accepted: Whether the plane passed the check. A numpy ``bool_`` is
                stored as ``bool``.

        Raises:
            TypeError: If ``plane`` is not a `SupportPlane`,
                ``gravity_angle_deg`` is not a real number, or ``accepted`` is
                not a boolean.
            ValueError: If ``gravity_angle_deg`` is not finite.
        """
        super().__init__()
        check_instance("plane", plane, SupportPlane)
        self._gravity_angle_deg = check_float("gravity_angle_deg", gravity_angle_deg)
        self._accepted = check_bool("accepted", accepted)
        self._plane = plane

    @property
    def plane(self) -> SupportPlane:
        """The checked support plane."""
        return self._plane

    @property
    def gravity_angle_deg(self) -> float:
        """Angle between the plane normal and the direction opposing gravity."""
        return self._gravity_angle_deg

    @property
    def accepted(self) -> bool:
        """Whether the plane passed the gravity check."""
        return self._accepted


def check_bool(name: str, value: Any) -> bool:
    """Return a ``bool`` or numpy ``bool_`` as a built-in ``bool``.

    Args:
        name: Field name used in error messages.
        value: Value to check.

    Returns:
        The value as a ``bool``.

    Raises:
        TypeError: If the value is not a ``bool`` or numpy ``bool_``.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return bool(value)
