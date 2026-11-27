"""Per-frame box measurement results."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .box_clouds import BoxViewDetection, _check_detections
from .collection import Collection


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


def _check_int(name: str, value: Any) -> int:
    """Return an integral number, ``bool`` excluded, as a built-in ``int``.

    Raises:
        TypeError: If the value is a ``bool`` or not a ``numbers.Integral``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _check_instance(name: str, value: Any, expected: type) -> None:
    """Check that a value is an instance of the expected type.

    Raises:
        TypeError: If the value is not an instance of ``expected``.
    """
    if not isinstance(value, expected):
        raise TypeError(
            f"{name} must be {expected.__name__}, got {type(value).__name__}"
        )


@dataclass(frozen=True, kw_only=True)
class BoxDimensions:
    """A box's length, width and height in millimeters.

    Attributes:
        length_mm: Length in millimeters.
        width_mm: Width in millimeters.
        height_mm: Height in millimeters.

    Raises:
        TypeError: If a field is not a real number.
        ValueError: If a field is not finite.
    """

    length_mm: float
    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        """Validate the fields and store them as built-in floats.

        Raises:
            TypeError: If a field is not a real number.
            ValueError: If a field is not finite.
        """
        for name in ("length_mm", "width_mm", "height_mm"):
            object.__setattr__(self, name, _check_float(name, getattr(self, name)))

    @property
    def volume_mm3(self) -> float:
        """The box volume in cubic millimeters."""
        return self.length_mm * self.width_mm * self.height_mm


class MeasurementStatus(str, Enum):
    """Whether a box was measured in a frame.

    ``INELIGIBLE`` boxes were not measured because no support plane is set.
    ``FAILED`` boxes were not measured because none of the box's views has enough
    depth coverage, or because its geometry could not be measured. ``MEASURED`` boxes
    produced a sample and have dimensions.
    """

    INELIGIBLE = "ineligible"
    FAILED = "failed"
    MEASURED = "measured"


@dataclass(frozen=True, kw_only=True)
class BoxMeasurementResult:
    """A box's measurement outcome in one processed frame.

    Attributes:
        box_id: The box's identity, kept across views and over time.
        detections: The 2D detection of each box view that took part in
            association this frame, at least one. A view may appear more than
            once.
        measurement_status: Outcome of measuring the box in this frame.
        dimensions: The box's aggregated dimensions, when it has any.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``detections`` is empty, or a measured box has no
            dimensions.
    """

    box_id: int
    detections: tuple[BoxViewDetection, ...]
    measurement_status: MeasurementStatus
    dimensions: BoxDimensions | None = None

    def __post_init__(self) -> None:
        """Validate the fields and store numbers as built-in types.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If ``detections`` is empty, or a measured box has no
                dimensions.
        """
        object.__setattr__(self, "box_id", _check_int("box_id", self.box_id))
        _check_detections(self.detections)
        _check_instance(
            "measurement_status", self.measurement_status, MeasurementStatus
        )
        if self.dimensions is not None:
            _check_instance("dimensions", self.dimensions, BoxDimensions)
        if (
            self.measurement_status is MeasurementStatus.MEASURED
            and self.dimensions is None
        ):
            raise ValueError("A measured box requires dimensions")


class BoxMeasurements(Collection[BoxMeasurementResult]):
    """Measurement results of the boxes in one frame.

    Args:
        items: Measurement results.

    Raises:
        TypeError: If ``items`` is not a list of one type.
    """
