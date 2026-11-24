"""Pair tracklets with segmented detections into one BoxDetections per frame."""

import depthai as dai
import numpy as np

from depthai_nodes.message import BoxDetection, BoxDetections
from depthai_nodes.message.box_detections import BACKGROUND_LABEL

from .segmentation_matching import match_tracklets_to_detections

# Tracker statuses whose box comes from the current frame's detections.
CURRENT_STATUSES = (
    dai.Tracklet.TrackingStatus.NEW,
    dai.Tracklet.TrackingStatus.TRACKED,
)
# Fraction of the frame; larger detections are false positives on the whole scene.
MAX_BBOX_AREA = 0.5
# Fraction of the frame size, on each side; a mask reaching into it is truncated.
BORDER_MARGIN = 0.05


def _is_truncated(region: np.ndarray) -> bool:
    """Return whether a non-empty mask region has a pixel centre in the margin."""
    height, width = region.shape
    rows = np.flatnonzero(region.any(axis=1))
    cols = np.flatnonzero(region.any(axis=0))
    return any(
        (first + 0.5) / size <= BORDER_MARGIN
        or (last + 0.5) / size >= 1 - BORDER_MARGIN
        for first, last, size in (
            (rows[0], rows[-1], height),
            (cols[0], cols[-1], width),
        )
    )


def _is_eligible(box: dai.RotatedRect, region: np.ndarray) -> bool:
    """Return whether a detection is non-empty, not whole-scene and not truncated."""
    return (
        bool(region.any())
        and box.size.width * box.size.height <= MAX_BBOX_AREA
        and not _is_truncated(region)
    )


class BoxDetectionAssembler(dai.node.HostNode):
    """Emit one box detection per new or tracked box with a non-empty segmentation.

    Whole-scene and truncated detections are left out.

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
            detections: The tracker's ``passthroughDetections``: the
                ``dai.ImgDetections`` it matched, with their instance mask.
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
        """Pair tracklets with detections and relabel the mask by item index."""
        source_mask = detections.getCvSegmentationMask()
        if source_mask is None or source_mask.ndim != 2:
            width, height = self._mask_size
            mask = np.full((height, width), BACKGROUND_LABEL, dtype=np.uint8)
            return BoxDetections([], mask)

        mask = np.full(source_mask.shape, BACKGROUND_LABEL, dtype=np.uint8)
        items: list[BoxDetection] = []
        # A lost track keeps its last-seen box, so pairing it could hand a
        # current detection, and its measurements, to the wrong track.
        active = [t for t in tracklets.tracklets if t.status in CURRENT_STATUSES]
        matches = match_tracklets_to_detections(active, detections.detections)
        # Emit in the parser's order, so mask IDs keep it too.
        for track_index, det_index in sorted(matches.items(), key=lambda m: m[1]):
            track, det = active[track_index], detections.detections[det_index]
            region = source_mask == det_index
            # The network input is stretch-resized, so normalized boxes already
            # use the source frame's coordinates.
            box = dai.RotatedRect(
                dai.Rect(
                    dai.Point2f(det.xmin, det.ymin), dai.Point2f(det.xmax, det.ymax)
                )
            )
            if not _is_eligible(box, region):
                continue
            if len(items) >= BACKGROUND_LABEL:
                break
            mask[region] = len(items)
            items.append(
                BoxDetection(
                    track_id=int(track.id),
                    bounding_box=box,
                    confidence=det.confidence,
                    track_status=track.status,
                )
            )
        return BoxDetections(items, mask)
