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


class ParcelFrameOutcome(str, Enum):
    """Whether a whole source frame completed processing."""

    SUCCESS = "success"
    FAILED = "failed"


class ParcelMeasurementStatus(str, Enum):
    """A parcel's footprint and height outcome."""

    NO_FIT = "no_fit"
    HEIGHT_UNKNOWN = "height_unknown"
    MEASURED = "measured"


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


@dataclass(frozen=True, kw_only=True)
class ParcelCameraCalibration:
    """Calibration for the exact delivered image pixel grid.

    Attributes:
        role: ``color``, ``left``, or ``right``.
        size: Delivered ``(width, height)`` in pixels.
        intrinsics: Full 3 by 3 matrix for this grid, including crop or resize.
        distortion_model: ``none``, ``perspective``, or ``fisheye``.
        distortion_coefficients: OpenCV coefficients for the delivered pixels.
        scene_to_camera_mm: Homogeneous 4 by 4 transform from the scene frame
            into this camera's optical frame. Translation is in millimeters.
    """

    role: Literal["color", "left", "right"]
    size: tuple[int, int]
    intrinsics: tuple[tuple[float, ...], ...]
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    scene_to_camera_mm: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.role not in ("color", "left", "right"):
            raise ValueError("Unknown camera role")
        if len(self.size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
            for v in self.size
        ):
            raise ValueError("size must be a positive integer (width, height)")
        intrinsics = _numbers(self.intrinsics, (3, 3), "intrinsics")
        transform = _numbers(self.scene_to_camera_mm, (4, 4), "scene transform")
        if not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-9, rtol=0):
            raise ValueError("Invalid homogeneous scene transform")
        if np.linalg.det(transform[:3, :3]) == 0 or np.linalg.det(intrinsics) == 0:
            raise ValueError("Singular calibration")
        coefficients = _numbers(
            self.distortion_coefficients,
            (len(self.distortion_coefficients),),
            "distortion coefficients",
        )
        model = self.distortion_model.casefold()
        valid = (
            (model == "none" and not coefficients.any())
            or (model == "perspective" and coefficients.size in (0, 4, 5, 8, 12, 14))
            or (model == "fisheye" and coefficients.size == 4)
        )
        if not valid:
            raise ValueError("Unsupported distortion model or coefficient count")
        object.__setattr__(self, "size", tuple(map(int, self.size)))
        object.__setattr__(self, "intrinsics", tuple(map(tuple, intrinsics.tolist())))
        object.__setattr__(
            self, "scene_to_camera_mm", tuple(map(tuple, transform.tolist()))
        )
        object.__setattr__(
            self, "distortion_coefficients", tuple(coefficients.tolist())
        )
        object.__setattr__(self, "distortion_model", model)


@dataclass(frozen=True, kw_only=True)
class ParcelRigCalibration:
    """The scene frame, downward gravity, and all three delivered camera grids.

    Attributes:
        frame: Coordinate-frame ID of both XYZ points and the supplied plane.
        down: Unit gravity vector in that frame. It determines upward plane sign.
        cameras: Exactly one color, left, and right calibration. Stored in that order.
    """

    frame: str
    down: tuple[float, float, float]
    cameras: tuple[ParcelCameraCalibration, ...]

    def __post_init__(self) -> None:
        _frame(self.frame)
        down = _numbers(self.down, (3,), "downward gravity")
        if abs(float(np.linalg.norm(down)) - 1) > 1e-6:
            raise ValueError("down must be a unit gravity vector")
        cameras = tuple(self.cameras)
        if any(not isinstance(c, ParcelCameraCalibration) for c in cameras):
            raise TypeError("cameras must contain ParcelCameraCalibration values")
        by_role = {c.role: c for c in cameras}
        if len(cameras) != 3 or set(by_role) != {"color", "left", "right"}:
            raise ValueError("Expected one color, left, and right camera")
        object.__setattr__(self, "down", tuple(down.tolist()))
        object.__setattr__(
            self, "cameras", tuple(by_role[r] for r in ("color", "left", "right"))
        )


class ParcelPlane(dai.Buffer):
    """A supplied support plane for exactly one source sequence number.

    Stamp this message with the cloud's host/device timestamps and sequence.
    ``plane=None`` explicitly reports that this frame has no supplied plane.
    Reusing a previous plane requires emitting a newly stamped message for the
    new frame. Plane estimation and plane selection remain upstream.
    """

    def __init__(self, *, frame: str, plane: SupportPlane | None) -> None:
        super().__init__()
        self._frame = _frame(frame)
        if plane is not None and not isinstance(plane, SupportPlane):
            raise TypeError("plane must be SupportPlane or None")
        self._plane = plane

    @property
    def frame(self) -> str:
        """Coordinate frame shared with the source cloud."""
        return self._frame

    @property
    def plane(self) -> SupportPlane | None:
        """Supplied plane, before orientation against scene gravity."""
        return self._plane


@dataclass(frozen=True, kw_only=True)
class ParcelSourceTiming:
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
        object.__setattr__(self, "sequence_num", int(self.sequence_num))


@dataclass(frozen=True, kw_only=True, eq=False)
class Parcel:
    """One eligible parcel with independent read-only float32 XYZ in millimeters.

    ``candidate_id`` is the zero-based proposal index within this frame. Gaps
    identify rejected proposals. Construction snapshots caller-owned points.
    """

    candidate_id: int
    xyz_mm: NDArray[np.float32]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _candidate(self.candidate_id))
        array = np.array(self.xyz_mm, dtype=np.float32, copy=True)
        self._check_points(array)
        array.flags.writeable = False
        object.__setattr__(self, "xyz_mm", array)

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
        result = object.__new__(cls)
        object.__setattr__(result, "candidate_id", _candidate(candidate_id))
        object.__setattr__(result, "xyz_mm", xyz_mm)
        return result


@dataclass(frozen=True, kw_only=True)
class ParcelDecision:
    """Optional proposal diagnostic, independent of measurement inputs."""

    candidate_id: int
    decision: Literal["eligible", "background", "partially_visible", "uncertain"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _candidate(self.candidate_id))
        if not isinstance(self.decision, str) or self.decision not in (
            "eligible",
            "background",
            "partially_visible",
            "uncertain",
        ):
            raise ValueError("Unknown eligibility decision")


@dataclass(frozen=True, kw_only=True)
class ParcelCuboid:
    """A parcel footprint and optional height, without temporal aggregation.

    The base center is on the support plane. The unit length axis is tangent to it. The
    width axis is the upward plane normal crossed with the length axis. All dimensions
    and the center use millimeters.
    """

    base_center_mm: tuple[float, float, float]
    length_axis: tuple[float, float, float]
    length_mm: float
    width_mm: float
    height_mm: float | None

    def __post_init__(self) -> None:
        center = _numbers(self.base_center_mm, (3,), "base center")
        axis = _numbers(self.length_axis, (3,), "length axis")
        if abs(float(axis @ axis) - 1) > 1e-6:
            raise ValueError("length_axis must be a unit vector")
        for name in ("length_mm", "width_mm", "height_mm"):
            value = getattr(self, name)
            if name == "height_mm" and value is None:
                continue
            number = float(_numbers(value, (), name))
            if not 1e-6 <= number <= 100_000:
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, number)
        if (np.abs(center) > 10_000_000).any():
            raise ValueError("Base center exceeds coordinate bounds")
        object.__setattr__(self, "base_center_mm", tuple(center.tolist()))
        object.__setattr__(self, "length_axis", tuple(axis.tolist()))


@dataclass(frozen=True, kw_only=True)
class ParcelMeasurement:
    """One candidate's no-fit, unknown-height, or measured cuboid outcome."""

    candidate_id: int
    status: ParcelMeasurementStatus
    cuboid: ParcelCuboid | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _candidate(self.candidate_id))
        if not isinstance(self.status, ParcelMeasurementStatus):
            raise TypeError("status must be ParcelMeasurementStatus")
        if self.status == ParcelMeasurementStatus.NO_FIT:
            if self.cuboid is not None:
                raise ValueError("A no-fit outcome cannot contain a cuboid")
        elif not isinstance(self.cuboid, ParcelCuboid):
            raise TypeError("A fitted outcome requires a ParcelCuboid")
        elif (self.cuboid.height_mm is None) != (
            self.status == ParcelMeasurementStatus.HEIGHT_UNKNOWN
        ):
            raise ValueError("Height must agree with the measurement status")


class _ParcelBatch(dai.Buffer):
    def __init__(
        self,
        *,
        items: tuple,
        frame: str,
        plane: SupportPlane | None,
        sources: tuple[ParcelSourceTiming, ...] = (),
        algorithm_id: str = ALGORITHM_ID,
        outcome: ParcelFrameOutcome = ParcelFrameOutcome.SUCCESS,
        failure_code: str | None = None,
    ) -> None:
        super().__init__()
        self._frame = _frame(frame)
        if plane is not None and not isinstance(plane, SupportPlane):
            raise TypeError("plane must be SupportPlane or None")
        if not isinstance(outcome, ParcelFrameOutcome):
            raise TypeError("outcome must be ParcelFrameOutcome")
        if not isinstance(algorithm_id, str) or not algorithm_id:
            raise ValueError("algorithm_id must be nonempty")
        items = tuple(items)
        if outcome == ParcelFrameOutcome.SUCCESS:
            if plane is None or failure_code is not None:
                raise ValueError("A successful frame requires a plane and no failure")
        elif items or not isinstance(failure_code, str) or not failure_code:
            raise ValueError("A failed frame requires no items and a failure code")
        ids = [item.candidate_id for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate IDs must be unique within a frame")
        sources = tuple(sources)
        if any(not isinstance(s, ParcelSourceTiming) for s in sources):
            raise TypeError("sources must contain ParcelSourceTiming values")
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
    def sources(self) -> tuple[ParcelSourceTiming, ...]:
        """Timing of every source that was available for this frame."""
        return self._sources

    @property
    def algorithm_id(self) -> str:
        """Full evaluated candidate identity."""
        return self._algorithm_id

    @property
    def outcome(self) -> ParcelFrameOutcome:
        """Whole-frame processing result."""
        return self._outcome

    @property
    def failure_code(self) -> str | None:
        """Stable failure reason, or None for a completed frame."""
        return self._failure_code


class ParcelDetections(_ParcelBatch):
    """One completed or failed frame containing only eligible parcel clouds.

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
        decisions: tuple[ParcelDecision, ...] = (),
        sources: tuple[ParcelSourceTiming, ...] = (),
        algorithm_id: str = ALGORITHM_ID,
        outcome: ParcelFrameOutcome = ParcelFrameOutcome.SUCCESS,
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
        if any(not isinstance(d, ParcelDecision) for d in decisions):
            raise TypeError("decisions must contain ParcelDecision values")
        if len({d.candidate_id for d in decisions}) != len(decisions):
            raise ValueError("Diagnostic candidate IDs must be unique")
        if decisions and {
            d.candidate_id for d in decisions if d.decision == "eligible"
        } != {p.candidate_id for p in items}:
            raise ValueError("Diagnostics must agree with eligible parcels")
        self._decisions = decisions

    @property
    def decisions(self) -> tuple[ParcelDecision, ...]:
        """Diagnostics for every proposed candidate, in proposal order."""
        return self._decisions

    @property
    def items(self) -> tuple[Parcel, ...]:
        """Eligible parcels in proposal order."""
        return self._items


class ParcelMeasurements(_ParcelBatch):
    """One measurement frame preserving detector identity, plane, and source timing.

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
        sources: tuple[ParcelSourceTiming, ...] = (),
        algorithm_id: str = ALGORITHM_ID,
        outcome: ParcelFrameOutcome = ParcelFrameOutcome.SUCCESS,
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
