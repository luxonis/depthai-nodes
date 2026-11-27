"""Box clouds of boxes associated across box views."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import depthai as dai
import numpy as np

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
class BoxViewDetection:
    """One box view's 2D detection.

    Attributes:
        view: Index of the view in ``BoxAssociator.build``'s views.
        track_id: That view's tracker ID.
        bounding_box: Normalized bounding box in that view's frame.
        confidence: Detection confidence.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``view`` is negative or ``confidence`` is not finite.
    """

    view: int
    track_id: int
    bounding_box: dai.RotatedRect
    confidence: float

    def __post_init__(self) -> None:
        """Validate the fields and store numbers as built-in types.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If ``view`` is negative or ``confidence`` is not finite.
        """
        view = _check_int("view", self.view)
        if view < 0:
            raise ValueError(f"view must be non-negative, got {view}")
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "track_id", _check_int("track_id", self.track_id))
        _check_instance("bounding_box", self.bounding_box, dai.RotatedRect)
        object.__setattr__(
            self, "confidence", _check_float("confidence", self.confidence)
        )


def _check_points(points: Any) -> None:
    """Check that ``points`` is an ``(N, 3)`` ``float32`` array.

    Raises:
        TypeError: If ``points`` is not a numpy array.
        ValueError: If ``points`` is not 2-D with three columns and ``float32``.
    """
    _check_instance("points", points, np.ndarray)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.float32:
        raise ValueError(
            "points must be an (N, 3) float32 array, "
            f"got shape {points.shape} and dtype {points.dtype}"
        )


def _check_detections(detections: Any) -> None:
    """Check that ``detections`` is a non-empty tuple of ``BoxViewDetection``.

    Raises:
        TypeError: If ``detections`` is not a tuple of ``BoxViewDetection``.
        ValueError: If ``detections`` is empty.
    """
    _check_instance("detections", detections, tuple)
    for detection in detections:
        _check_instance("detections item", detection, BoxViewDetection)
    if not detections:
        raise ValueError("detections must not be empty")


@dataclass(frozen=True, kw_only=True, eq=False)
class BoxCloud:
    """One box's box cloud in one frame.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``points`` is not an ``(N, 3)`` ``float32`` array,
            ``depth_coverage`` is not finite and in ``[0, 1]``, or
            ``detections`` is empty.
    """

    box_id: int
    """The box's identity across views and frames.

    It is never reused within a run, and it is unrelated to any ``track_id``.
    """

    points: np.ndarray
    """An ``(N, 3)`` ``float32`` array in millimetres, in shared coordinates."""

    depth_coverage: float
    """The highest depth coverage among the box's box views, between 0 and 1."""

    detections: tuple[BoxViewDetection, ...]
    """One ``BoxViewDetection`` per box view that took part in association.

    Never empty and in no promised order. One view may appear more than once.
    """

    def __post_init__(self) -> None:
        """Validate the fields and store numbers as built-in types.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field has an invalid value.
        """
        object.__setattr__(self, "box_id", _check_int("box_id", self.box_id))
        _check_points(self.points)
        depth_coverage = _check_float("depth_coverage", self.depth_coverage)
        if not 0.0 <= depth_coverage <= 1.0:
            raise ValueError(f"depth_coverage must be in [0, 1], got {depth_coverage}")
        object.__setattr__(self, "depth_coverage", depth_coverage)
        _check_detections(self.detections)


class BoxClouds(Collection[BoxCloud]):
    """Box clouds of one frame, one per box seen in it.

    Args:
        items: Box clouds.

    Raises:
        TypeError: If ``items`` is not a list of ``BoxCloud``.
    """

    def __init__(self, items: list[BoxCloud]):
        """Initialize the collection, accepting only ``BoxCloud`` items.

        Args:
            items: Box clouds.

        Raises:
            TypeError: If ``items`` is not a list of ``BoxCloud``.
        """
        super().__init__([])
        self._item_cls = BoxCloud
        self.items = items
