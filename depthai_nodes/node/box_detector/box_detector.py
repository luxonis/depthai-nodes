"""Tracked box detections from a color stream, with an internally owned model."""

import math
import numbers
from typing import Any, cast

import depthai as dai

from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode

from .box_detection_assembler import BoxDetectionAssembler

MODEL = "luxonis/box-instance-segmentation:512x320"


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


def _check_threshold(value: Any) -> float:
    """Return a confidence threshold as a ``float``.

    Args:
        value: Threshold to check.

    Returns:
        The threshold as a ``float``.

    Raises:
        TypeError: If the threshold is a ``bool`` or not a real number.
        ValueError: If the threshold is not finite or not between zero and one.
    """
    threshold = _check_float("confidenceThreshold", value)
    if not 0 <= threshold <= 1:
        raise ValueError(f"confidenceThreshold must be in [0, 1], got {threshold}")
    return threshold


class BoxDetector(BaseThreadedHostNode):
    """Detect, segment and track boxes in a color stream on RVC4.

    The node runs the ``luxonis/box-instance-segmentation`` model from the Model
    Zoo and tracks its detections. The confidence threshold is set at build and
    can change while the pipeline runs through `setConfidenceThreshold`.

    The model and the tracker run on the device that produces the color stream. A
    color stream produced on the host uses the pipeline's default device. In a
    pipeline without an implicit device, add the devices before creating the node.

    Attributes:
        out: Output stream of `depthai_nodes.message.BoxDetections`: one per
            processed frame, holding a box detection for every new or tracked box
            with a current, non-empty segmentation, and carrying the frame's
            timestamps and sequence number. Detections covering more than half of
            the frame (false positives on the whole scene) and truncated boxes,
            whose segmentation reaches the frame border, are left out.
    """

    def __init__(self) -> None:
        """Create the node and its detection assembler."""
        super().__init__()
        self._parser: dai.node.DetectionParser | None = None
        self._config_queue: dai.InputQueue | None = None
        self._assembler = self.createSubnode(BoxDetectionAssembler)

    def build(
        self, color: dai.Node.Output, confidenceThreshold: float = 0.5
    ) -> "BoxDetector":
        """Build the detection, tracking and assembly subnodes.

        Args:
            color: Color stream to detect boxes in. Its producing device runs the
                model and the tracker.
            confidenceThreshold: Minimum detection confidence, between zero and one.

        Returns:
            The configured node instance.

        Raises:
            RuntimeError: If the node is already built.
            TypeError: If ``confidenceThreshold`` is not a real number.
            ValueError: If ``confidenceThreshold`` is not finite or not between
                zero and one, if the color stream's device is not an RVC4, or if the
                model archive has no input size.
        """
        if self._parser is not None:
            raise RuntimeError("BoxDetector is already built")
        threshold = _check_threshold(confidenceThreshold)
        source = color.getParent()
        if source.runOnHost():
            device = self.getParentPipeline().getDefaultDevice()
        else:
            # A node that doesn't run on the host is a device node.
            device = cast(dai.DeviceNode, source).getDevice()
        platform = device.getPlatform()
        if platform != dai.Platform.RVC4:
            raise ValueError("BoxDetector currently requires an RVC4 device")
        description = dai.NNModelDescription(MODEL, platform=platform.name)
        archive = dai.NNArchive(dai.getModelFromZoo(description))
        size = archive.getInputSize()
        if size is None:
            raise ValueError("NNArchive input size is None")

        resize = self.createSubnode(dai.node.ImageManip, device)
        resize.initialConfig.setOutputSize(
            *size, dai.ImageManipConfig.ResizeMode.STRETCH
        )
        resize.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
        color.link(resize.inputImage)
        # The parser owns the threshold: decoding low-confidence candidates is slow.
        nn = self.createSubnode(dai.node.DetectionNetwork, device).build(
            resize.out, archive, confidenceThreshold=threshold
        )
        parser = nn.detectionParser
        self._config_queue = parser.inputConfig.createInputQueue()
        self._parser = parser

        tracker = self.createSubnode(dai.node.ObjectTracker, device)
        tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
        nn.out.link(tracker.inputDetections)
        color.link(tracker.inputTrackerFrame)
        self._assembler.build(tracker.out, tracker.passthroughDetections, size)
        return self

    @property
    def out(self) -> dai.Node.Output:
        """Return the stream of tracked ``BoxDetections``."""
        return self._assembler.out

    def setConfidenceThreshold(self, confidenceThreshold: float) -> None:
        """Change the minimum detection confidence while the pipeline runs.

        The parser's IoU threshold is kept.

        Args:
            confidenceThreshold: Minimum detection confidence, between zero and one.

        Raises:
            RuntimeError: If the node is not built yet.
            TypeError: If ``confidenceThreshold`` is not a real number.
            ValueError: If ``confidenceThreshold`` is not finite or not between
                zero and one.
        """
        if self._parser is None or self._config_queue is None:
            raise RuntimeError("BoxDetector is not built")
        threshold = _check_threshold(confidenceThreshold)
        config = dai.DetectionParserConfig()
        config.setConfidenceThreshold(threshold)
        config.setIouThreshold(self._parser.getIouThreshold())
        self._config_queue.send(config)

    def run(self) -> None:
        """Do nothing: the subnodes do the work."""
        return
