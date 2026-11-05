"""Portable support plane geometry, independent of running pipeline nodes."""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True, kw_only=True)
class SupportPlane:
    """The normalized plane equation ``normal · point + offset_mm = 0``.

    Coordinates are in millimeters. Construction scales the normal to unit
    length and the offset with it, so equations that differ only by a positive
    factor give equal planes.

    Attributes:
        normal: Unit normal ``(x, y, z)``.
        offset_mm: Offset in millimeters.

    Raises:
        TypeError: If ``normal`` is not a list or tuple, or a component is not a
            real number.
        ValueError: If ``normal`` does not have three components, is zero, or a
            value is not finite.
    """

    normal: tuple[float, float, float]
    offset_mm: float

    def __post_init__(self) -> None:
        """Validate and normalize the plane equation.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field has an invalid value.
        """
        if not isinstance(self.normal, (list, tuple)):
            raise TypeError("Support plane normal must be a list or tuple")
        if len(self.normal) != 3:
            raise ValueError("Support plane normal must have three components")
        normal = tuple(_check_float("normal", value) for value in self.normal)
        offset = _check_float("offset_mm", self.offset_mm)
        # Scaling first keeps normalization safe for very large finite normals.
        scale = max(abs(value) for value in normal)
        if scale == 0:
            raise ValueError("Support plane normal must be nonzero")
        scaled = tuple(value / scale for value in normal)
        length = math.hypot(*scaled)
        normalized_offset = (offset / scale) / length
        if not math.isfinite(normalized_offset):
            raise ValueError("Normalized support plane offset must be finite")
        object.__setattr__(self, "normal", tuple(value / length for value in scaled))
        object.__setattr__(self, "offset_mm", normalized_offset)

    def toJson(self) -> dict[str, Any]:
        """Return the normalized geometry as a JSON-compatible dictionary.

        Returns:
            A dictionary with exactly ``normal`` (a list) and ``offset_mm``.
        """
        return {"normal": list(self.normal), "offset_mm": self.offset_mm}

    @classmethod
    def fromJson(cls, data: dict[str, Any]) -> SupportPlane:
        """Create a plane from a dictionary written by ``toJson``.

        Args:
            data: Dictionary with exactly ``normal`` and ``offset_mm``.

        Returns:
            The normalized support plane.

        Raises:
            TypeError: If ``data`` is not a dictionary or holds non-numeric
                geometry.
            ValueError: If the keys differ from ``normal`` and ``offset_mm``, or
                the geometry is invalid.
        """
        if not isinstance(data, dict):
            raise TypeError("Expected a support plane JSON object")
        if set(data) != {"normal", "offset_mm"}:
            raise ValueError("Expected exactly support plane normal and offset_mm")
        return cls(normal=data["normal"], offset_mm=data["offset_mm"])

    def toJsonFile(self, path: str | Path) -> None:
        """Write the plane as JSON to a file.

        Args:
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        Path(path).write_text(
            json.dumps(self.toJson(), allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def fromJsonFile(cls, path: str | Path) -> SupportPlane:
        """Read a plane from a JSON file written by ``toJsonFile``.

        Args:
            path: Source file path.

        Returns:
            The normalized support plane.

        Raises:
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file is not valid JSON.
            TypeError: If the JSON has the wrong types.
            ValueError: If the JSON has the wrong keys or invalid geometry.
        """
        return cls.fromJson(json.loads(Path(path).read_text(encoding="utf-8")))
