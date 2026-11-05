"""Per-frame box detections, before any measurement."""

import math
import numbers
from dataclasses import dataclass
from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.utils import AnnotationHelper

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


BACKGROUND_LABEL = 255


@dataclass(frozen=True, kw_only=True)
class BoxDetection:
    """A tracked box's identity and location in one processed frame.

    Its segmentation is the region of the owning `BoxDetections` mask labelled
    with this item's index.

    Attributes:
        track_id: Tracker ID of the box.
        bounding_box: Normalized bounding box in the source frame.
        confidence: Detection confidence.
        track_status: Tracking status of the box in this frame.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``confidence`` is not finite.
    """

    track_id: int
    bounding_box: dai.RotatedRect
    confidence: float
    track_status: dai.Tracklet.TrackingStatus

    def __post_init__(self) -> None:
        """Validate the fields and store numbers as built-in types.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If ``confidence`` is not finite.
        """
        object.__setattr__(self, "track_id", _check_int("track_id", self.track_id))
        _check_instance("bounding_box", self.bounding_box, dai.RotatedRect)
        object.__setattr__(
            self, "confidence", _check_float("confidence", self.confidence)
        )
        _check_instance("track_status", self.track_status, dai.Tracklet.TrackingStatus)


class BoxDetections(Collection[BoxDetection]):
    """Box detections sharing one segmentation label mask.

    The mask covers the whole source frame, stretched to its own resolution. A
    pixel value is the index of the item it belongs to; 255 is background.

    The mask is checked against the items only at construction. Assigning
    ``items``, ``append()`` and ``extend()`` do not recheck it.

    Args:
        items: Detections, indexed by the mask labels.
        segmentation_mask: 2-D ``uint8`` label mask.

    Raises:
        TypeError: If ``segmentation_mask`` is not a numpy array, or ``items``
            is not a list of one type.
        ValueError: If ``segmentation_mask`` is not a 2-D ``uint8`` array, or a
            label does not index an item.
    """

    def __init__(self, items: list[BoxDetection], segmentation_mask: np.ndarray):
        """Initialize the detections and check the mask against them.

        Args:
            items: Detections, indexed by the mask labels.
            segmentation_mask: 2-D ``uint8`` label mask; 255 is background.

        Raises:
            TypeError: If ``segmentation_mask`` is not a numpy array, or
                ``items`` is not a list of one type.
            ValueError: If ``segmentation_mask`` is not a 2-D ``uint8`` array,
                or a label does not index an item.
        """
        super().__init__(items)
        if not isinstance(segmentation_mask, np.ndarray):
            raise TypeError("Segmentation mask must be a numpy array")
        if segmentation_mask.ndim != 2 or segmentation_mask.dtype != np.uint8:
            raise ValueError("Segmentation mask must be a 2-D uint8 label image")
        if items and segmentation_mask.size == 0:
            raise ValueError("Segmentation mask must not be empty when items exist")
        labels = segmentation_mask[segmentation_mask != BACKGROUND_LABEL]
        if labels.size and int(labels.max()) >= len(items):
            raise ValueError("Segmentation mask labels must index detection items")
        self._segmentation_mask = segmentation_mask

    @property
    def segmentation_mask(self) -> np.ndarray:
        """The shared ``uint8`` label mask; 255 marks background."""
        return self._segmentation_mask

    def getMask(self, index: int) -> np.ndarray:
        """Return the segmentation mask of one detection.

        Args:
            index: Index of the detection in ``items``.

        Returns:
            Boolean mask, ``True`` where the pixel belongs to the detection.

        Raises:
            IndexError: If ``index`` is outside ``items``.
        """
        if not 0 <= index < len(self.items):
            raise IndexError("Detection index out of range")
        return self._segmentation_mask == index

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        """Outline each box, labelled with its track ID and confidence.

        Returns:
            Annotations with this message's timestamps and sequence number.
        """
        annotations = AnnotationHelper()
        for box in self.items:
            xmin, ymin, xmax, ymax = box.bounding_box.getOuterRect()
            annotations.draw_rectangle(
                (xmin, ymin), (xmax, ymax), fill_color=None, thickness=3.0
            )
            annotations.draw_text(
                f"Box {box.track_id} ({box.confidence:.0%})",
                (xmin, max(0.0, ymin - 0.01)),
                color=(1.0, 1.0, 1.0, 1.0),
                background_color=(0.0, 0.0, 0.0, 0.6),
                size=28.0,
            )
        message = annotations.build(self.getTimestamp(), self.getSequenceNum())
        message.setTimestampDevice(self.getTimestampDevice())
        return message
