import depthai as dai
import numpy as np

from depthai_nodes.message import (
    BoxDetection,
    BoxDetections,
    BoxDimensions,
    BoxMeasurementResult,
    BoxMeasurements,
    MeasurementStatus,
)

BOX_DIMENSIONS = BoxDimensions(length_mm=200, width_mm=100, height_mm=50)


def create_rotated_rect(
    center: tuple[float, float] = (0.5, 0.5),
    size: tuple[float, float] = (0.2, 0.2),
) -> dai.RotatedRect:
    """Create a zero-angle normalized rectangle for tests.

    Args:
        center: Normalized ``(x, y)`` center.
        size: Normalized ``(width, height)``.

    Returns:
        ``dai.RotatedRect``: Rectangle with the supplied geometry.
    """
    return dai.RotatedRect(dai.Point2f(*center), dai.Size2f(*size), 0)


def create_box_detection(
    track_id: int = 7,
    center: tuple[float, float] = (0.5, 0.5),
    confidence: float = 0.8,
    track_status: dai.Tracklet.TrackingStatus = dai.Tracklet.TrackingStatus.TRACKED,
) -> BoxDetection:
    """Create a box detection with a 0.2 by 0.2 bounding box for tests.

    Args:
        track_id: Tracker ID.
        center: Normalized bounding box center.
        confidence: Detection confidence.
        track_status: Tracking status.

    Returns:
        ``BoxDetection``: Detection populated with the supplied values.
    """
    return BoxDetection(
        track_id=track_id,
        bounding_box=create_rotated_rect(center),
        confidence=confidence,
        track_status=track_status,
    )


def create_box_detections(
    items: list[BoxDetection] | None = None,
    segmentation_mask: np.ndarray | None = None,
) -> BoxDetections:
    """Create box detections for tests.

    Args:
        items: Detections; defaults to none.
        segmentation_mask: Label mask; defaults to an all-background 4 by 6 mask.

    Returns:
        ``BoxDetections``: Message holding the items and mask.
    """
    if segmentation_mask is None:
        segmentation_mask = np.full((4, 6), 255, dtype=np.uint8)
    return BoxDetections(items or [], segmentation_mask)


def create_box_measurement_result(
    measurement_status: MeasurementStatus = MeasurementStatus.MEASURED,
    dimensions: BoxDimensions | None = BOX_DIMENSIONS,
    track_id: int = 7,
) -> BoxMeasurementResult:
    """Create a box measurement result with a 0.4 by 0.4 bounding box for tests.

    Args:
        measurement_status: Measurement status.
        dimensions: Aggregated dimensions, if any.
        track_id: Tracker ID.

    Returns:
        ``BoxMeasurementResult``: Result populated with the supplied values.
    """
    return BoxMeasurementResult(
        track_id=track_id,
        bounding_box=create_rotated_rect(size=(0.4, 0.4)),
        confidence=0.8,
        track_status=dai.Tracklet.TrackingStatus.TRACKED,
        measurement_status=measurement_status,
        dimensions=dimensions,
    )


def create_box_measurements(
    items: list[BoxMeasurementResult] | None = None,
) -> BoxMeasurements:
    """Create box measurements for tests.

    Args:
        items: Measurement results; defaults to none.

    Returns:
        ``BoxMeasurements``: Message holding the items.
    """
    return BoxMeasurements(items or [])
