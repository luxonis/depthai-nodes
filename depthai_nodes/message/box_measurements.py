"""Per-frame box measurement results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import depthai as dai

from depthai_nodes.utils import AnnotationHelper

from .collection import Collection
from .utils.value_checks import check_float, check_instance, check_int


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
            object.__setattr__(self, name, check_float(name, getattr(self, name)))

    @property
    def volume_mm3(self) -> float:
        """The box volume in cubic millimeters."""
        return self.length_mm * self.width_mm * self.height_mm


class MeasurementStatus(str, Enum):
    """Whether a box was measured in a frame.

    ``INELIGIBLE`` boxes could not be measured, for example because no support
    plane is set or the box touches the frame border. ``FAILED`` boxes were
    eligible but their geometry could not be measured. ``MEASURED`` boxes
    produced a sample and have dimensions.
    """

    INELIGIBLE = "ineligible"
    FAILED = "failed"
    MEASURED = "measured"


@dataclass(frozen=True, kw_only=True)
class BoxMeasurementResult:
    """A tracked box's measurement outcome in one processed frame.

    Attributes:
        track_id: Tracker ID of the box.
        bounding_box: Normalized bounding box in the source frame.
        confidence: Detection confidence.
        track_status: Tracking status of the box in this frame.
        measurement_status: Outcome of measuring the box in this frame.
        dimensions: The box's aggregated dimensions, when it has any.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``confidence`` is not finite, or a measured box has no
            dimensions.
    """

    track_id: int
    bounding_box: dai.RotatedRect
    confidence: float
    track_status: dai.Tracklet.TrackingStatus
    measurement_status: MeasurementStatus
    dimensions: BoxDimensions | None = None

    def __post_init__(self) -> None:
        """Validate the fields and store numbers as built-in types.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If ``confidence`` is not finite, or a measured box has
                no dimensions.
        """
        object.__setattr__(self, "track_id", check_int("track_id", self.track_id))
        check_instance("bounding_box", self.bounding_box, dai.RotatedRect)
        object.__setattr__(
            self, "confidence", check_float("confidence", self.confidence)
        )
        check_instance("track_status", self.track_status, dai.Tracklet.TrackingStatus)
        check_instance("measurement_status", self.measurement_status, MeasurementStatus)
        if self.dimensions is not None:
            check_instance("dimensions", self.dimensions, BoxDimensions)
        if (
            self.measurement_status is MeasurementStatus.MEASURED
            and self.dimensions is None
        ):
            raise ValueError("A measured box requires dimensions")


class BoxMeasurements(Collection[BoxMeasurementResult]):
    """Measurement results of the boxes tracked in one frame.

    Args:
        items: Measurement results.

    Raises:
        TypeError: If ``items`` is not a list of one type.
    """

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        """Outline each box, labelled with its identity and any dimensions.

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
                annotation_text(box),
                (xmin, max(0.0, ymin - 0.01)),
                color=(1.0, 1.0, 1.0, 1.0),
                background_color=(0.0, 0.0, 0.0, 0.6),
                size=28.0,
            )
        message = annotations.build(self.getTimestamp(), self.getSequenceNum())
        message.setTimestampDevice(self.getTimestampDevice())
        return message


def annotation_text(box: BoxMeasurementResult) -> str:
    """Return the annotation text of a measured box.

    Args:
        box: Measurement result to describe.

    Returns:
        Track ID and confidence, followed by the dimensions in centimeters and
        the volume in cubic centimeters when the box has dimensions.
    """
    text = f"Box {box.track_id} ({box.confidence:.0%})"
    dimensions = box.dimensions
    if dimensions is not None:
        text += (
            f"\nH: {dimensions.height_mm / 10:.1f} cm"
            f"\nL: {dimensions.length_mm / 10:.1f} cm"
            f" W: {dimensions.width_mm / 10:.1f} cm"
            f"\nV: {dimensions.volume_mm3 / 1000:.0f} cm³"
        )
    return text
