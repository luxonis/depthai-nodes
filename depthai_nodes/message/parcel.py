"""Per-frame parcel geometry and same-host Python messages.

Arrays and nested Python values are retained only within one host runtime. XLink and
RemoteConnection serialization is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Literal

import depthai as dai
import numpy as np
from numpy.typing import NDArray

from depthai_nodes.algorithms.parcel import ALGORITHM_ID

from .support_plane import SupportPlane

__all__ = [
    "Parcel",
    "ParcelCuboid",
    "ParcelDetections",
    "ParcelMeasurement",
    "ParcelMeasurements",
    "ParcelPlane",
]


def _frame(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("A nonempty coordinate frame ID is required")
    return value


def _candidate(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("candidate_id must be an integer")
    if value < 0:
        raise ValueError("candidate_id must be nonnegative")
    return int(value)


def _numbers(value, shape: tuple[int, ...], field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {field}")
    return result


class ParcelPlane(dai.Buffer):
    """A supplied support plane for exactly one source sequence number.

    Stamp this message with the cloud's host/device timestamps and sequence.
    ``plane=None`` explicitly reports that this frame has no supplied plane.
    Reusing a previous plane requires emitting a newly stamped message for the
    new frame. ``down`` supplies unit gravity in the cloud frame; image metadata
    does not determine gravity. Plane estimation and selection remain upstream.
    """

    def __init__(
        self,
        *,
        frame: str,
        plane: SupportPlane | None,
        down: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self._frame = _frame(frame)
        if plane is not None and not isinstance(plane, SupportPlane):
            raise TypeError("plane must be SupportPlane or None")
        gravity = _numbers(down, (3,), "downward gravity")
        if abs(float(np.linalg.norm(gravity)) - 1) > 1e-6:
            raise ValueError("down must be a unit gravity vector")
        self._plane = plane
        self._down = tuple(gravity.tolist())

    @property
    def down(self) -> tuple[float, float, float]:
        """Unit downward gravity in the point cloud and support-plane frame."""
        return self._down

    @property
    def frame(self) -> str:
        """Coordinate frame shared with the source cloud."""
        return self._frame

    @property
    def plane(self) -> SupportPlane | None:
        """Supplied plane, before orientation against scene gravity."""
        return self._plane


@dataclass(frozen=True, kw_only=True)
class _ParcelSourceTiming:
    """Host/device timestamps and sequence of one source stream."""

    stream: Literal["point_cloud", "color", "left", "right", "support_plane"]
    timestamp: timedelta
    timestamp_device: timedelta
    sequence_num: int

    def __post_init__(self) -> None:
        if self.stream not in (
            "point_cloud",
            "color",
            "left",
            "right",
            "support_plane",
        ):
            raise ValueError("Unknown source stream")
        if not isinstance(self.timestamp, timedelta) or not isinstance(
            self.timestamp_device, timedelta
        ):
            raise TypeError("Source timestamps must be datetime.timedelta values")
        if isinstance(self.sequence_num, bool) or not isinstance(
            self.sequence_num, (int, np.integer)
        ):
            raise TypeError("Source sequence number must be an integer")


class Parcel:
    """One eligible parcel with read-only float32 XYZ in millimeters.

    ``candidate_id`` is the zero-based proposal index within this frame. Gaps
    identify rejected proposals. Construction retains a read-only owning
    float32 array and snapshots other inputs into independent point storage.
    """

    def __init__(self, *, candidate_id: int, xyz_mm: NDArray[np.float32]) -> None:
        self._candidate_id = _candidate(candidate_id)
        array = np.asarray(xyz_mm, dtype=np.float32)
        self._check_points(array)
        if not array.flags.owndata or array.flags.writeable:
            array = array.copy()
        array.flags.writeable = False
        self._xyz_mm = array

    @property
    def candidate_id(self) -> int:
        """Zero-based proposal index within this frame."""
        return self._candidate_id

    @property
    def xyz_mm(self) -> NDArray[np.float32]:
        """Retained read-only point array in selected-point order."""
        return self._xyz_mm

    @staticmethod
    def _check_points(array: np.ndarray) -> None:
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError("Parcel points must be finite (N, 3) XYZ")

    @classmethod
    def _from_owned(cls, candidate_id: int, xyz_mm: NDArray[np.float32]) -> Parcel:
        cls._check_points(xyz_mm)
        if xyz_mm.dtype != np.float32 or not xyz_mm.flags.owndata:
            raise ValueError("Extracted parcel must own float32 point storage")
        xyz_mm.flags.writeable = False
        return cls(candidate_id=candidate_id, xyz_mm=xyz_mm)


@dataclass(frozen=True, kw_only=True)
class _ParcelDecision:
    """Optional proposal diagnostic, independent of measurement inputs."""

    candidate_id: int
    decision: Literal["eligible", "background", "partially_visible", "uncertain"]

    def __post_init__(self) -> None:
        _candidate(self.candidate_id)
        if not isinstance(self.decision, str) or self.decision not in (
            "eligible",
            "background",
            "partially_visible",
            "uncertain",
        ):
            raise ValueError("Unknown eligibility decision")


class ParcelCuboid:
    """A parcel footprint and optional height, without temporal aggregation.

    The base center is on the support plane. The unit length axis is tangent to it. The
    width axis is the upward plane normal crossed with the length axis. All dimensions
    and the center use millimeters. Construction snapshots the vectors and validates the
    finite geometry.
    """

    def __init__(
        self,
        *,
        base_center_mm: tuple[float, float, float],
        length_axis: tuple[float, float, float],
        length_mm: float,
        width_mm: float,
        height_mm: float | None,
    ) -> None:
        center = _numbers(base_center_mm, (3,), "base center")
        axis = _numbers(length_axis, (3,), "length axis")
        if abs(float(axis @ axis) - 1) > 1e-6:
            raise ValueError("length_axis must be a unit vector")
        if (np.abs(center) > 10_000_000).any():
            raise ValueError("Base center exceeds coordinate bounds")
        self._base_center_mm = tuple(center.tolist())
        self._length_axis = tuple(axis.tolist())
        self._length_mm = self._dimension(length_mm, "length_mm")
        self._width_mm = self._dimension(width_mm, "width_mm")
        self._height_mm = (
            None if height_mm is None else self._dimension(height_mm, "height_mm")
        )

    @staticmethod
    def _dimension(value: float, name: str) -> float:
        number = float(_numbers(value, (), name))
        if not 1e-6 <= number <= 100_000:
            raise ValueError(f"Invalid {name}")
        return number

    @property
    def base_center_mm(self) -> tuple[float, float, float]:
        """Cuboid base center on the support plane in millimeters."""
        return self._base_center_mm

    @property
    def length_axis(self) -> tuple[float, float, float]:
        """Unit length axis tangent to the support plane."""
        return self._length_axis

    @property
    def length_mm(self) -> float:
        """Cuboid length in millimeters."""
        return self._length_mm

    @property
    def width_mm(self) -> float:
        """Cuboid width in millimeters."""
        return self._width_mm

    @property
    def height_mm(self) -> float | None:
        """Cuboid height in millimeters, or None when unknown."""
        return self._height_mm


@dataclass(frozen=True, kw_only=True)
class ParcelMeasurement:
    """One candidate's no-fit, unknown-height, or measured cuboid outcome.

    ``Status`` identifies the footprint and height result.
    """

    class Status(str, Enum):
        """A parcel's footprint and height outcome."""

        NO_FIT = "no_fit"
        HEIGHT_UNKNOWN = "height_unknown"
        MEASURED = "measured"

    candidate_id: int
    status: ParcelMeasurement.Status
    cuboid: ParcelCuboid | None

    def __post_init__(self) -> None:
        _candidate(self.candidate_id)
        if not isinstance(self.status, self.Status):
            raise TypeError("status must be ParcelMeasurement.Status")
        if self.status == self.Status.NO_FIT:
            if self.cuboid is not None:
                raise ValueError("A no-fit outcome cannot contain a cuboid")
        elif not isinstance(self.cuboid, ParcelCuboid):
            raise TypeError("A fitted outcome requires a ParcelCuboid")
        elif (self.cuboid.height_mm is None) != (
            self.status == self.Status.HEIGHT_UNKNOWN
        ):
            raise ValueError("Height must agree with the measurement status")


class _ParcelBatch(dai.Buffer):
    class Outcome(str, Enum):
        """Whether a whole source frame completed processing.

        Both parcel batch messages expose the same enum as ``Outcome``.
        """

        SUCCESS = "success"
        FAILED = "failed"

    def __init__(
        self,
        *,
        items: tuple,
        frame: str,
        plane: SupportPlane | None,
        sources: tuple[_ParcelSourceTiming, ...] = (),
        algorithm_id: str = ALGORITHM_ID,
        outcome: _ParcelBatch.Outcome = Outcome.SUCCESS,
        failure_code: str | None = None,
    ) -> None:
        super().__init__()
        self._frame = _frame(frame)
        if plane is not None and not isinstance(plane, SupportPlane):
            raise TypeError("plane must be SupportPlane or None")
        if not isinstance(outcome, self.Outcome):
            raise TypeError("outcome must be a parcel batch Outcome")
        if not isinstance(algorithm_id, str) or not algorithm_id:
            raise ValueError("algorithm_id must be nonempty")
        items = tuple(items)
        if outcome == self.Outcome.SUCCESS:
            if plane is None or failure_code is not None:
                raise ValueError("A successful frame requires a plane and no failure")
        elif items or not isinstance(failure_code, str) or not failure_code:
            raise ValueError("A failed frame requires no items and a failure code")
        ids = [item.candidate_id for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate IDs must be unique within a frame")
        sources = tuple(sources)
        if any(not isinstance(s, _ParcelSourceTiming) for s in sources):
            raise TypeError("sources must contain source timing records")
        if len({s.stream for s in sources}) != len(sources):
            raise ValueError("Source stream timings must be unique")
        self._items = items
        self._plane = plane
        self._sources = sources
        self._algorithm_id = algorithm_id
        self._outcome = outcome
        self._failure_code = failure_code
        for source in sources:
            if source.stream == "point_cloud":
                self.setTimestamp(source.timestamp)
                self.setTimestampDevice(source.timestamp_device)
                self.setSequenceNum(source.sequence_num)

    @property
    def items(self) -> tuple:
        """Immutable per-frame items in proposal order."""
        return self._items

    @property
    def frame(self) -> str:
        """Coordinate-frame ID shared by the points and plane."""
        return self._frame

    @property
    def plane(self) -> SupportPlane | None:
        """Exact upward plane snapshot used by the detector."""
        return self._plane

    @property
    def sources(self) -> tuple[_ParcelSourceTiming, ...]:
        """Timing of every source that was available for this frame."""
        return self._sources

    @property
    def algorithm_id(self) -> str:
        """Full evaluated candidate identity."""
        return self._algorithm_id

    @property
    def outcome(self) -> _ParcelBatch.Outcome:
        """Whole-frame processing result."""
        return self._outcome

    @property
    def failure_code(self) -> str | None:
        """Stable failure reason, or None for a completed frame."""
        return self._failure_code


class ParcelDetections(_ParcelBatch):
    """One completed or failed frame containing only eligible parcel clouds.

    ``Outcome`` distinguishes successful and failed frames.

    Items own their points until consumption. No source cloud or image is retained.
    ``decisions`` contains proposal diagnostics without source-point indices.
    Native Buffer timestamps and sequence repeat the source cloud metadata.
    """

    def __init__(
        self,
        *,
        frame: str,
        plane: SupportPlane | None,
        items: tuple[Parcel, ...] = (),
        decisions: tuple[_ParcelDecision, ...] = (),
        sources: tuple[_ParcelSourceTiming, ...] = (),
        algorithm_id: str = ALGORITHM_ID,
        outcome: _ParcelBatch.Outcome = _ParcelBatch.Outcome.SUCCESS,
        failure_code: str | None = None,
    ) -> None:
        items = tuple(items)
        if any(not isinstance(item, Parcel) for item in items):
            raise TypeError("items must contain Parcel values")
        super().__init__(
            items=items,
            frame=frame,
            plane=plane,
            sources=sources,
            algorithm_id=algorithm_id,
            outcome=outcome,
            failure_code=failure_code,
        )
        decisions = tuple(decisions)
        if any(not isinstance(d, _ParcelDecision) for d in decisions):
            raise TypeError("decisions must contain diagnostic decision records")
        if len({d.candidate_id for d in decisions}) != len(decisions):
            raise ValueError("Diagnostic candidate IDs must be unique")
        if decisions and {
            d.candidate_id for d in decisions if d.decision == "eligible"
        } != {p.candidate_id for p in items}:
            raise ValueError("Diagnostics must agree with eligible parcels")
        self._decisions = decisions

    @property
    def decisions(self) -> tuple[_ParcelDecision, ...]:
        """Diagnostics for every proposed candidate, in proposal order."""
        return self._decisions

    @property
    def items(self) -> tuple[Parcel, ...]:
        """Eligible parcels in proposal order."""
        return self._items


class ParcelMeasurements(_ParcelBatch):
    """One measurement frame preserving detector identity, plane, and source timing.

    ``Outcome`` is shared with ``ParcelDetections.Outcome``.

    Each eligible parcel has one outcome, including no-fit and unknown-height. An empty
    successful input produces an empty successful output. A failed input or measurement
    call produces a failed frame with no partial results.
    """

    def __init__(
        self,
        *,
        frame: str,
        plane: SupportPlane | None,
        items: tuple[ParcelMeasurement, ...] = (),
        sources: tuple[_ParcelSourceTiming, ...] = (),
        algorithm_id: str = ALGORITHM_ID,
        outcome: _ParcelBatch.Outcome = _ParcelBatch.Outcome.SUCCESS,
        failure_code: str | None = None,
    ) -> None:
        items = tuple(items)
        if any(not isinstance(item, ParcelMeasurement) for item in items):
            raise TypeError("items must contain ParcelMeasurement values")
        super().__init__(
            items=items,
            frame=frame,
            plane=plane,
            sources=sources,
            algorithm_id=algorithm_id,
            outcome=outcome,
            failure_code=failure_code,
        )

    @property
    def items(self) -> tuple[ParcelMeasurement, ...]:
        """Per-parcel geometry outcomes in input order."""
        return self._items
