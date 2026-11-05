"""Pair tracklets with segmented detections into one BoxDetections per frame."""

import depthai as dai
import numpy as np

from depthai_nodes.message import BoxDetection, BoxDetections
from depthai_nodes.message.box_detections import BACKGROUND_LABEL

from .segmentation_matching import match_tracklets_to_detections


class BoxDetectionAssembler(dai.node.HostNode):
    """Emit one box detection per tracked box with a non-empty segmentation.

    Attributes:
        out: Output stream of ``BoxDetections`` carrying the timestamps and sequence
            number of the input detections.
    """

    def __init__(self) -> None:
        """Initialize the assembler with a placeholder mask size."""
        super().__init__()
        self._mask_size = (1, 1)

    def build(
        self,
        tracklets: dai.Node.Output,
        detections: dai.Node.Output,
        maskSize: tuple[int, int],
    ) -> "BoxDetectionAssembler":
        """Link the tracker and detection streams.

        Args:
            tracklets: Output producing ``dai.Tracklets``.
            detections: Output producing ``dai.ImgDetections`` with an instance mask.
            maskSize: ``(width, height)`` of the network input, used for the mask of
                frames that carry none.

        Returns:
            This node.
        """
        self._mask_size = maskSize
        self.link_args(tracklets, detections)
        return self

    def process(self, tracklets: dai.Buffer, detections: dai.Buffer) -> None:
        """Assemble and emit the box detections of one frame.

        Args:
            tracklets: Tracker output for the frame.
            detections: Detections of the frame with their instance mask.

        Raises:
            TypeError: If the inputs are not ``dai.Tracklets`` and
                ``dai.ImgDetections``.
        """
        if not isinstance(tracklets, dai.Tracklets) or not isinstance(
            detections, dai.ImgDetections
        ):
            raise TypeError("Expected dai.Tracklets and dai.ImgDetections")
        message = self._assemble(tracklets, detections)
        message.setTimestamp(detections.getTimestamp())
        message.setTimestampDevice(detections.getTimestampDevice())
        message.setSequenceNum(detections.getSequenceNum())
        try:
            self.out.send(message)
        # DepthAI's stub declares QueueException through an undeclared nested
        # class, so pyright cannot tell it is an exception; at runtime it is one.
        except dai.MessageQueue.QueueException:  # pyright: ignore[reportGeneralTypeIssues]
            return

    def _assemble(
        self, tracklets: dai.Tracklets, detections: dai.ImgDetections
    ) -> BoxDetections:
        source_mask = detections.getCvSegmentationMask()
        if source_mask is None or source_mask.ndim != 2:
            width, height = self._mask_size
            mask = np.full((height, width), BACKGROUND_LABEL, dtype=np.uint8)
            return BoxDetections([], mask)

        mask = np.full(source_mask.shape, BACKGROUND_LABEL, dtype=np.uint8)
        items: list[BoxDetection] = []
        active = [
            t
            for t in tracklets.tracklets
            if t.status != dai.Tracklet.TrackingStatus.REMOVED
        ]
        matches = match_tracklets_to_detections(active, detections.detections)
        # Emit in the parser's order, so mask IDs keep it too.
        for track_index, det_index in sorted(matches.items(), key=lambda m: m[1]):
            track, det = active[track_index], detections.detections[det_index]
            region = source_mask == det_index
            if not region.any() or len(items) >= BACKGROUND_LABEL:
                continue
            mask[region] = len(items)
            # The network input is stretch-resized, so normalized boxes already
            # use the source frame's coordinates.
            corners = dai.Rect(
                dai.Point2f(det.xmin, det.ymin), dai.Point2f(det.xmax, det.ymax)
            )
            items.append(
                BoxDetection(
                    track_id=int(track.id),
                    bounding_box=dai.RotatedRect(corners),
                    confidence=det.confidence,
                    track_status=track.status,
                )
            )
        return BoxDetections(items, mask)
