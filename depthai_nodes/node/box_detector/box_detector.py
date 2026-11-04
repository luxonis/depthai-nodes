"""Tracked box detections from a color stream, with an internally owned model."""

from typing import Any

import depthai as dai

from depthai_nodes.message.utils.value_checks import check_float
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.img_detections_filter import ImgDetectionsFilter
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork

from .box_detection_assembler import BoxDetectionAssembler

MODEL = "luxonis/box-instance-segmentation:512x320"


def check_threshold(value: Any) -> float:
    """Return a confidence threshold as a ``float``.

    Args:
        value: Threshold to check.

    Returns:
        The threshold as a ``float``.

    Raises:
        TypeError: If the threshold is a ``bool`` or not a real number.
        ValueError: If the threshold is not finite or not between zero and one.
    """
    threshold = check_float("confidenceThreshold", value)
    if not 0 <= threshold <= 1:
        raise ValueError(f"confidenceThreshold must be in [0, 1], got {threshold}")
    return threshold


class BoxDetector(BaseThreadedHostNode):
    """Detect, segment and track boxes in a color stream on RVC4.

    The node runs the ``luxonis/box-instance-segmentation`` model from the Model
    Zoo and tracks its detections. The confidence threshold can change while the
    pipeline runs through `setConfidenceThreshold`.

    Attributes:
        out: Output stream of `depthai_nodes.message.BoxDetections`: one per
            processed frame, holding a box detection for every tracked box with a
            current, non-empty segmentation, and carrying the frame's timestamps
            and sequence number.
    """

    def __init__(self) -> None:
        """Create the node and its detection assembler.

        Raises:
            ValueError: If the pipeline has no default device.
        """
        try:
            super().__init__()
        except AttributeError:
            # The depthai-nodes base reads the default device's platform.
            if self.getParentPipeline().getDefaultDevice() is not None:
                raise
            raise ValueError("BoxDetector currently requires an RVC4 device") from None
        self._filter: ImgDetectionsFilter | None = None
        self._assembler = self.createSubnode(BoxDetectionAssembler)

    def build(
        self, color: dai.Node.Output, confidenceThreshold: float = 0.5
    ) -> "BoxDetector":
        """Build the detection, filtering and tracking subnodes.

        Args:
            color: Color stream to detect boxes in.
            confidenceThreshold: Minimum detection confidence, between zero and one.

        Returns:
            This node.

        Raises:
            RuntimeError: If the node is already built.
            TypeError: If ``confidenceThreshold`` is not a real number.
            ValueError: If ``confidenceThreshold`` is not finite or not between
                zero and one, if the device is not an RVC4, or if the model archive
                has no input size.
        """
        if self._filter is not None:
            raise RuntimeError("BoxDetector is already built")
        threshold = check_threshold(confidenceThreshold)
        if self._platform != dai.Platform.RVC4:
            raise ValueError("BoxDetector currently requires an RVC4 device")
        description = dai.NNModelDescription(MODEL, platform=self._platform.name)
        archive = dai.NNArchive(dai.getModelFromZoo(description))
        size = archive.getInputSize()
        if size is None:
            raise ValueError("NNArchive input size is None")

        resize = self.createSubnode(dai.node.ImageManip)
        resize.initialConfig.setOutputSize(
            *size, dai.ImageManipConfig.ResizeMode.STRETCH
        )
        resize.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
        color.link(resize.inputImage)
        nn = self.createSubnode(ParsingNeuralNetwork).build(resize.out, archive)

        # Parser at 0 and threshold in a filter: live parser config needs depthai 3.11.
        nn.getParser(dai.node.DetectionParser).setConfidenceThreshold(0)
        detection_filter = self.createSubnode(ImgDetectionsFilter).build(nn.out)
        detection_filter.minConfidence(threshold)

        tracker = self.createSubnode(dai.node.ObjectTracker)
        tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
        detection_filter.out.link(tracker.inputDetections)
        color.link(tracker.inputTrackerFrame)
        self._assembler.build(tracker.out, detection_filter.out, size)
        self._filter = detection_filter
        return self

    @property
    def out(self) -> dai.Node.Output:
        """Return the stream of tracked ``BoxDetections``."""
        return self._assembler.out

    def setConfidenceThreshold(self, confidenceThreshold: float) -> None:
        """Change the minimum detection confidence while the pipeline runs.

        Args:
            confidenceThreshold: Minimum detection confidence, between zero and one.

        Raises:
            RuntimeError: If the node is not built yet.
            TypeError: If ``confidenceThreshold`` is not a real number.
            ValueError: If ``confidenceThreshold`` is not finite or not between
                zero and one.
        """
        if self._filter is None:
            raise RuntimeError("BoxDetector is not built")
        self._filter.minConfidence(check_threshold(confidenceThreshold))

    def run(self) -> None:
        """Do nothing: the subnodes do the work."""
        return
