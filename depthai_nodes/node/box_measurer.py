"""Box measurement from views, composed of detection, association and dimensioning."""

from collections.abc import Sequence
from datetime import timedelta

import depthai as dai

from depthai_nodes.message.support_plane import SupportPlane
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.box_associator import BoxAssociator
from depthai_nodes.node.box_associator.box_associator import (
    MAX_VIEWS,
    _check_sync_threshold,
)
from depthai_nodes.node.box_detector import BoxDetector
from depthai_nodes.node.box_detector.box_detector import _check_threshold
from depthai_nodes.node.box_dimensioner import BoxDimensioner
from depthai_nodes.node.box_dimensioner.box_dimensioner import _check_instance


class BoxMeasurer(BaseThreadedHostNode):
    """Detect, track and measure boxes seen by one or more views on RVC4.

    A view is one device's color stream and the depth aligned to it, covering the
    same field of view at any resolution. Each view's detector runs on the device
    that produces its color stream. Box views from different views are associated
    into one box per physical box, which keeps one box identity across views and
    over time.

    Points and the support plane are in shared coordinates. With several views, the
    pipeline's multi-device calibration defines them; with one, they are the
    depth's own frame. With several views, views are grouped by host timestamp, and
    a frame whose views are not captured within the sync threshold produces no
    output. The default threshold (depthai's 10 ms) needs frame-synced capture
    (PTP).

    Measurement stops for good when any view's device is lost or any view's stream
    stalls, while the pipeline keeps running. Call
    ``pipeline.setStopOnDeviceLoss(True)`` to stop the pipeline instead.

    In a pipeline without an implicit device, add the devices before creating the
    node. Requires open3d, installed with
    ``pip install "depthai-nodes[box-measurement]"``.

    Attributes:
        out: Output stream of `depthai_nodes.message.BoxMeasurements`, one result
            per box in the frame.
    """

    def __init__(self) -> None:
        """Create the association and dimensioning subnodes."""
        super().__init__()
        self._built = False
        self._detectors: list[BoxDetector] = []
        self._associator = self.createSubnode(BoxAssociator)
        self._dimensioner = self.createSubnode(BoxDimensioner)

    def build(
        self,
        views: Sequence[tuple[dai.Node.Output, dai.Node.Output]],
        supportPlane: SupportPlane | None = None,
        confidenceThreshold: float = 0.5,
        syncThreshold: timedelta | None = None,
    ) -> "BoxMeasurer":
        """Build one detector per view, the associator and the dimensioner.

        A failed ``build`` leaves the node unusable.

        Args:
            views: One ``(color, depth)`` pair per view. ``color`` is the stream to
                detect boxes in. ``depth`` is the depth aligned to ``color``: it
                covers the same field of view, at any resolution. Depth that isn't
                aligned to its color isn't detected; boxes are then labelled at the
                wrong points. The views are fixed at build.
            supportPlane: Initial active support plane; ``None`` leaves boxes
                ineligible until `setSupportPlane` is called.
            confidenceThreshold: Minimum detection confidence, between zero and one.
            syncThreshold: The widest spread of host timestamps among the views of
                one frame. ``None`` keeps depthai's default of 10 ms, which needs
                frame-synced capture (PTP). Without PTP, use about half the frame
                period. A larger value can group frames that were captured one
                frame apart. Use a larger value only for static or slow scenes: a
                moving box is at a different position in each view, and it can be
                split into two boxes or measured too large.

        Returns:
            The configured node instance.

        Raises:
            ImportError: If open3d is not installed.
            RuntimeError: If ``build`` was already called, even if that call failed.
            TypeError: If ``supportPlane`` is not a ``SupportPlane``,
                ``confidenceThreshold`` is not a real number, or ``syncThreshold``
                is not ``None`` and not a ``timedelta``.
            ValueError: If ``views`` is empty or holds more than 255 views, if
                ``confidenceThreshold`` is not finite or not between zero and one,
                if ``syncThreshold`` is zero or negative, or if a view's color
                stream's device is not an RVC4.
        """
        if self._built:
            raise RuntimeError("BoxMeasurer.build was already called")
        if supportPlane is not None:
            _check_instance("supportPlane", supportPlane, SupportPlane)
        threshold = _check_threshold(confidenceThreshold)
        _check_sync_threshold(syncThreshold)
        if not 1 <= len(views) <= MAX_VIEWS:
            raise ValueError(
                f"views must hold 1 to {MAX_VIEWS} views, got {len(views)}"
            )
        self._built = True
        for color, _ in views:
            self._detectors.append(
                self.createSubnode(BoxDetector).build(color, threshold)
            )
        self._associator.build(
            [
                (detector.out, depth)
                for detector, (_, depth) in zip(self._detectors, views)
            ],
            syncThreshold,
        )
        self._dimensioner.build(self._associator.out, supportPlane)
        return self

    @property
    def out(self) -> dai.Node.Output:
        """Return the stream of `depthai_nodes.message.BoxMeasurements`."""
        return self._dimensioner.out

    def setSupportPlane(self, supportPlane: SupportPlane) -> None:
        """Measure against a support plane from the next frame.

        Works before and after build, also while the pipeline runs. Aggregated
        dimensions are reset.

        Args:
            supportPlane: The new active support plane.

        Raises:
            TypeError: If ``supportPlane`` is not a ``SupportPlane``.
        """
        self._dimensioner.setSupportPlane(supportPlane)

    def getSupportPlane(self) -> SupportPlane | None:
        """Return the active support plane.

        Returns:
            ``SupportPlane`` or ``None``: The active support plane, if set.
        """
        return self._dimensioner.getSupportPlane()

    def clearSupportPlane(self) -> None:
        """Stop measuring from the next frame.

        Aggregated dimensions are reset, and boxes are ineligible until a support plane
        is set again.
        """
        self._dimensioner.clearSupportPlane()

    def setConfidenceThreshold(self, confidenceThreshold: float) -> None:
        """Change every view's minimum detection confidence while the pipeline runs.

        Args:
            confidenceThreshold: Minimum detection confidence, between zero and one.

        Raises:
            RuntimeError: If the node is not built yet.
            TypeError: If ``confidenceThreshold`` is not a real number.
            ValueError: If ``confidenceThreshold`` is not finite or not between
                zero and one.
        """
        if not self._detectors:
            raise RuntimeError("BoxMeasurer is not built")
        for detector in self._detectors:
            detector.setConfidenceThreshold(confidenceThreshold)

    def run(self) -> None:
        """Do nothing: the subnodes do the work."""
        return
