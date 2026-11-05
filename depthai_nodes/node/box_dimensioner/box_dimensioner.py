import numbers
from collections import deque
from collections.abc import Sequence
from statistics import median
from threading import Lock
from typing import Any, cast

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.box_detections import BoxDetections
from depthai_nodes.message.box_measurements import (
    BoxDimensions,
    BoxMeasurementResult,
    BoxMeasurements,
    MeasurementStatus,
)
from depthai_nodes.message.support_plane import SupportPlane
from depthai_nodes.node.base_host_node import BaseHostNode

from .box_geometry import BoxGeometry


def _check_size(name: str, value: Any) -> tuple[int, int]:
    """Return a ``(width, height)`` sequence of positive integers as a tuple.

    Raises:
        TypeError: If the value is not a sequence, is a ``str`` or ``bytes``, or
            holds elements that are ``bool`` or not integers.
        ValueError: If the value does not have exactly two positive elements.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a (width, height) sequence, got {type(value).__name__}"
        )
    if len(value) != 2:
        raise ValueError(f"{name} must be (width, height), got {len(value)} values")
    for label, element in zip(("width", "height"), value):
        if isinstance(element, bool) or not isinstance(element, numbers.Integral):
            raise TypeError(
                f"{name} {label} must be an integer, got {type(element).__name__}"
            )
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive, got ({width}, {height})")
    return width, height


def _check_instance(name: str, value: Any, expected: type) -> None:
    """Check that a value is an instance of the expected type.

    Raises:
        TypeError: If the value is not an instance of ``expected``.
    """
    if not isinstance(value, expected):
        raise TypeError(
            f"{name} must be {expected.__name__}, got {type(value).__name__}"
        )


# Fraction of the frame; larger detections are false positives on the whole scene.
MAX_BBOX_AREA = 0.5
# Fraction of the frame; boxes this close to an edge may be cut off.
BORDER_MARGIN = 0.05
# Minimum fraction of mask pixels with a valid point for a box to be measured.
MIN_VALID_POINT_RATIO = 0.90
# Valid measurements per track that the aggregated dimensions are taken over.
MAX_SAMPLES_PER_TRACK = 600
# Consecutive absent frames after which a track's samples are discarded.
MAX_ABSENT_FRAMES = 30


class BoxDimensioner(BaseHostNode):
    """Measure box detections in an organized millimeter point cloud.

    Emits `depthai_nodes.message.BoxMeasurements` on ``out``, one result per box
    detection with dimensions aggregated over the track, and the valid points
    under all masks as ``dai.PointCloudData`` on ``boxCloudOutput``. Both carry
    the point cloud's timestamps and sequence number. The detections' source
    frame and the cloud must cover the same field of view; their resolutions may
    differ.

    Boxes are measured against the active support plane, set through ``build`` or
    `setSupportPlane`. Without one, every box is ineligible. Detections covering
    more than half of the frame are omitted. Requires open3d, installed with
    ``pip install depthai-nodes[box-measurement]``.
    """

    def __init__(self) -> None:
        """Create the measurement and box-cloud outputs."""
        super().__init__()
        self._box_cloud_output = self.createOutput(name="box_cloud")
        self._plane_lock = Lock()
        self._plane: SupportPlane | None = None
        self._plane_revision = 0
        self._seen_plane_revision = -1
        self._samples: dict[int, deque[BoxDimensions]] = {}
        self._absent: dict[int, int] = {}
        self._geometry = BoxGeometry()
        self._size: tuple[int, int] | None = None

    @property
    def boxCloudOutput(self) -> dai.Node.Output:
        """Valid points under all box masks, as ``dai.PointCloudData``.

        Returns:
            ``dai.Node.Output``: The box point-cloud output.
        """
        return self._box_cloud_output

    def build(
        self,
        detections: dai.Node.Output,
        pointCloud: dai.Node.Output,
        pointCloudSize: tuple[int, int],
        supportPlane: SupportPlane | None = None,
    ) -> "BoxDimensioner":
        """Link the inputs and set the cloud size and the support plane.

        Args:
            detections: Output emitting `depthai_nodes.message.BoxDetections`.
            pointCloud: Output emitting organized ``dai.PointCloudData`` in
                millimeters.
            pointCloudSize: Cloud ``(width, height)`` in points.
            supportPlane: Initial active support plane; ``None`` leaves boxes
                ineligible until `setSupportPlane` is called.

        Returns:
            The configured node instance.

        Raises:
            ImportError: If open3d is not installed.
            RuntimeError: If the node is already built.
            TypeError: If ``pointCloudSize`` is not a pair of integers or
                ``supportPlane`` is not a ``SupportPlane``.
            ValueError: If ``pointCloudSize`` does not have two positive values.
        """
        size = self._check_build(pointCloudSize, supportPlane)
        if supportPlane is not None:
            self.setSupportPlane(supportPlane)
        self._size = size
        self.link_args(detections, pointCloud)
        return self

    def _check_build(
        self, pointCloudSize: tuple[int, int], supportPlane: SupportPlane | None
    ) -> tuple[int, int]:
        """Raise what `build` would raise for these arguments, changing nothing.

        Returns:
            The checked cloud ``(width, height)``.
        """
        try:
            import open3d  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__} requires open3d. Install it with "
                "`pip install depthai-nodes[box-measurement]`."
            ) from e
        if self._size is not None:
            raise RuntimeError("BoxDimensioner is already built")
        size = _check_size("pointCloudSize", pointCloudSize)
        if supportPlane is not None:
            _check_instance("supportPlane", supportPlane, SupportPlane)
        return size

    def setSupportPlane(self, supportPlane: SupportPlane) -> None:
        """Measure against a support plane from the next frame.

        Aggregated dimensions are reset.

        Args:
            supportPlane: The new active support plane.

        Raises:
            TypeError: If ``supportPlane`` is not a ``SupportPlane``.
        """
        _check_instance("supportPlane", supportPlane, SupportPlane)
        with self._plane_lock:
            self._plane = supportPlane
            self._plane_revision += 1

    def getSupportPlane(self) -> SupportPlane | None:
        """Return the active support plane.

        Returns:
            ``SupportPlane`` or ``None``: The active support plane, if set.
        """
        with self._plane_lock:
            return self._plane

    def clearSupportPlane(self) -> None:
        """Stop measuring from the next frame.

        Aggregated dimensions are reset, and boxes are ineligible until a support plane
        is set again.
        """
        with self._plane_lock:
            self._plane = None
            self._plane_revision += 1

    def process(self, detections: dai.Buffer, pointCloud: dai.Buffer) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Measure one frame's box detections in its point cloud.

        Args:
            detections: The frame's `depthai_nodes.message.BoxDetections`.
            pointCloud: The frame's organized ``dai.PointCloudData``.

        Raises:
            TypeError: If the messages are not box detections and a point cloud.
        """
        try:
            self._process(detections, pointCloud)
        # DepthAI's stub declares QueueException through an undeclared nested
        # class, so pyright cannot tell it is an exception; at runtime it is one.
        except dai.MessageQueue.QueueException:  # pyright: ignore[reportGeneralTypeIssues]
            return

    def _process(self, detections: dai.Buffer, point_cloud: dai.Buffer) -> None:
        """Measure each detection against the active plane and send the results."""
        _check_instance("detections", detections, BoxDetections)
        _check_instance("pointCloud", point_cloud, dai.PointCloudData)
        detections = cast(BoxDetections, detections)
        point_cloud = cast(dai.PointCloudData, point_cloud)
        with self._plane_lock:
            plane, revision = self._plane, self._plane_revision
        if revision != self._seen_plane_revision:
            self._samples.clear()
            self._absent.clear()
            self._seen_plane_revision = revision
        assert self._size is not None
        width, height = self._size
        points = np.asarray(point_cloud.getPoints())
        organized = points.shape == (width * height, 3)
        if organized:
            points = points.reshape(height, width, 3)
        present: set[int] = set()
        results: list[BoxMeasurementResult] = []
        clouds: list[np.ndarray] = []
        for index, detection in enumerate(detections.items):
            track_id = detection.track_id
            present.add(track_id)
            self._absent[track_id] = 0
            rect = detection.bounding_box
            if rect.size.width * rect.size.height > MAX_BBOX_AREA:
                # Omitted, but still present, so its samples are kept.
                continue
            # The mask covers the whole source frame, as the cloud does, so
            # nearest-neighbor resizing maps it onto the cloud grid.
            mask = cv2.resize(
                detections.getMask(index).astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            xmin, ymin, xmax, ymax = rect.getOuterRect()
            border = (
                xmin <= BORDER_MARGIN
                or ymin <= BORDER_MARGIN
                or xmax >= 1 - BORDER_MARGIN
                or ymax >= 1 - BORDER_MARGIN
            )
            status = (
                MeasurementStatus.INELIGIBLE
                if plane is None or border
                else MeasurementStatus.FAILED
            )
            box_points = points[mask] if organized else np.empty((0, 3))
            valid = np.isfinite(box_points).all(axis=1) & (box_points[:, 2] > 1e-6)
            box_points = box_points[valid]
            if box_points.size:
                clouds.append(box_points)
            mask_pixels = int(mask.sum())
            if (
                status == MeasurementStatus.FAILED
                and mask_pixels > 0
                and len(box_points) / mask_pixels >= MIN_VALID_POINT_RATIO
            ):
                assert plane is not None
                try:
                    dimensions = self._geometry.measure(
                        box_points, np.array((*plane.normal, plane.offset_mm))
                    )
                except (ValueError, RuntimeError, cv2.error):
                    dimensions = None
                if dimensions is not None and all(
                    v > 0
                    for v in (
                        dimensions.length_mm,
                        dimensions.width_mm,
                        dimensions.height_mm,
                    )
                ):
                    self._samples.setdefault(
                        track_id, deque(maxlen=MAX_SAMPLES_PER_TRACK)
                    ).append(dimensions)
                    status = MeasurementStatus.MEASURED
            results.append(
                BoxMeasurementResult(
                    track_id=track_id,
                    bounding_box=rect,
                    confidence=detection.confidence,
                    track_status=detection.track_status,
                    measurement_status=status,
                    dimensions=self._aggregate(track_id),
                )
            )
        for track_id in list(self._absent):
            if track_id in present:
                continue
            self._absent[track_id] += 1
            if self._absent[track_id] >= MAX_ABSENT_FRAMES:
                self._absent.pop(track_id)
                self._samples.pop(track_id, None)
        out = BoxMeasurements(results)
        self._stamp(out, point_cloud)
        self.out.send(out)
        cloud = dai.PointCloudData()
        output_points = (np.concatenate(clouds) if clouds else np.empty((0, 3))).astype(
            np.float32
        )
        cloud.setPoints(output_points)  # pyright: ignore[reportArgumentType]
        self._stamp(cloud, point_cloud)
        self._box_cloud_output.send(cloud)

    @staticmethod
    def _stamp(message: dai.Buffer, source: dai.Buffer) -> None:
        """Copy the source's timestamps and sequence number onto a message."""
        message.setTimestamp(source.getTimestamp())
        message.setTimestampDevice(source.getTimestampDevice())
        message.setSequenceNum(source.getSequenceNum())

    def _aggregate(self, track_id: int) -> BoxDimensions | None:
        """Return the per-dimension median of a track's samples, if any."""
        samples = self._samples.get(track_id)
        if not samples:
            return None
        return BoxDimensions(
            length_mm=median(s.length_mm for s in samples),
            width_mm=median(s.width_mm for s in samples),
            height_mm=median(s.height_mm for s in samples),
        )
