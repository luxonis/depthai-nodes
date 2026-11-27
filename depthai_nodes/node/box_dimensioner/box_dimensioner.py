from collections import deque
from statistics import median
from threading import Lock
from typing import Any, cast

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.box_clouds import BoxCloud, BoxClouds
from depthai_nodes.message.box_measurements import (
    BoxDimensions,
    BoxMeasurementResult,
    BoxMeasurements,
    MeasurementStatus,
)
from depthai_nodes.message.support_plane import SupportPlane
from depthai_nodes.node.base_host_node import BaseHostNode

from .box_geometry import BoxGeometry


def _check_instance(name: str, value: Any, expected: type) -> None:
    """Check that a value is an instance of the expected type.

    Raises:
        TypeError: If the value is not an instance of ``expected``.
    """
    if not isinstance(value, expected):
        raise TypeError(
            f"{name} must be {expected.__name__}, got {type(value).__name__}"
        )


# The lowest ``BoxCloud.depth_coverage`` that is measured.
MIN_VALID_POINT_RATIO = 0.90
# Valid measurements per box ID that the aggregated dimensions are taken over.
MAX_SAMPLES_PER_BOX = 600
# Consecutive ``BoxClouds`` frames without a box ID after which its samples are
# dropped.
MAX_ABSENT_FRAMES = 30


class BoxDimensioner(BaseHostNode):
    """Measure box clouds against the active support plane.

    Emits `depthai_nodes.message.BoxMeasurements` on ``out``: one result per box
    cloud, with dimensions aggregated per box identity. The box clouds and the
    support plane must be in the same shared coordinates.

    The support plane is set through ``build`` or `setSupportPlane`. Without one,
    every box is ineligible.
    """

    def __init__(self) -> None:
        """Create the node without a support plane."""
        super().__init__()
        self._plane_lock = Lock()
        self._plane: SupportPlane | None = None
        self._plane_revision = 0
        self._seen_plane_revision = -1
        self._samples: dict[int, deque[BoxDimensions]] = {}
        self._absent: dict[int, int] = {}
        self._geometry = BoxGeometry()
        self._built = False

    def build(
        self, boxClouds: dai.Node.Output, supportPlane: SupportPlane | None = None
    ) -> "BoxDimensioner":
        """Link the box clouds and set the support plane.

        Args:
            boxClouds: Output emitting `depthai_nodes.message.BoxClouds`.
            supportPlane: Initial active support plane; ``None`` leaves boxes
                ineligible until `setSupportPlane` is called.

        Returns:
            The configured node instance.

        Raises:
            RuntimeError: If the node is already built.
            TypeError: If ``supportPlane`` is not a ``SupportPlane``.
        """
        if self._built:
            raise RuntimeError("BoxDimensioner is already built")
        if supportPlane is not None:
            self.setSupportPlane(supportPlane)
        self._built = True
        self.link_args(boxClouds)
        return self

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

    def process(self, boxClouds: dai.Buffer) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Measure one frame's box clouds.

        Args:
            boxClouds: The frame's `depthai_nodes.message.BoxClouds`.

        Raises:
            TypeError: If the message is not ``BoxClouds``.
        """
        try:
            self._process(boxClouds)
        # DepthAI's stub declares QueueException through an undeclared nested
        # class, so pyright cannot tell it is an exception; at runtime it is one.
        except dai.MessageQueue.QueueException:  # pyright: ignore[reportGeneralTypeIssues]
            return

    def _process(self, box_clouds: dai.Buffer) -> None:
        """Measure each box cloud against the active plane and send the results."""
        _check_instance("boxClouds", box_clouds, BoxClouds)
        box_clouds = cast(BoxClouds, box_clouds)
        with self._plane_lock:
            plane, revision = self._plane, self._plane_revision
        if revision != self._seen_plane_revision:
            self._samples.clear()
            self._absent.clear()
            self._seen_plane_revision = revision
        present: set[int] = set()
        results: list[BoxMeasurementResult] = []
        for cloud in box_clouds.items:
            present.add(cloud.box_id)
            self._absent[cloud.box_id] = 0
            status = self._measure_and_record(cloud, plane)
            results.append(
                BoxMeasurementResult(
                    box_id=cloud.box_id,
                    detections=cloud.detections,
                    measurement_status=status,
                    dimensions=self._aggregate(cloud.box_id),
                )
            )
        for box_id in list(self._absent):
            if box_id in present:
                continue
            self._absent[box_id] += 1
            if self._absent[box_id] >= MAX_ABSENT_FRAMES:
                self._absent.pop(box_id)
                self._samples.pop(box_id, None)
        out = BoxMeasurements(results)
        self._stamp(out, box_clouds)
        self.out.send(out)

    def _measure_and_record(
        self, cloud: BoxCloud, plane: SupportPlane | None
    ) -> MeasurementStatus:
        """Measure a box cloud, store a valid sample and return the status."""
        if plane is None:
            return MeasurementStatus.INELIGIBLE
        if cloud.depth_coverage < MIN_VALID_POINT_RATIO:
            return MeasurementStatus.FAILED
        try:
            dimensions = self._geometry.measure(
                cloud.points, np.array((*plane.normal, plane.offset_mm))
            )
        except (ValueError, RuntimeError, cv2.error):
            return MeasurementStatus.FAILED
        if dimensions is None or not all(
            v > 0
            for v in (dimensions.length_mm, dimensions.width_mm, dimensions.height_mm)
        ):
            return MeasurementStatus.FAILED
        self._samples.setdefault(
            cloud.box_id, deque(maxlen=MAX_SAMPLES_PER_BOX)
        ).append(dimensions)
        return MeasurementStatus.MEASURED

    @staticmethod
    def _stamp(message: dai.Buffer, source: dai.Buffer) -> None:
        """Copy the source's timestamps and sequence number onto a message."""
        message.setTimestamp(source.getTimestamp())
        message.setTimestampDevice(source.getTimestampDevice())
        message.setSequenceNum(source.getSequenceNum())

    def _aggregate(self, box_id: int) -> BoxDimensions | None:
        """Return the per-dimension median of a box's samples, if any."""
        samples = self._samples.get(box_id)
        if not samples:
            return None
        return BoxDimensions(
            length_mm=median(s.length_mm for s in samples),
            width_mm=median(s.width_mm for s in samples),
            height_mm=median(s.height_mm for s in samples),
        )
