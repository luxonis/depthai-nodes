"""Box measurement from color and a point cloud, composed of detection and
dimensioning."""

import depthai as dai

from depthai_nodes.message.support_plane import SupportPlane
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.box_detector import BoxDetector
from depthai_nodes.node.box_dimensioner import BoxDimensioner


class BoxMeasurer(BaseThreadedHostNode):
    """Detect, track and measure boxes from color and a point cloud on RVC4.

    Wires a `BoxDetector` into a `BoxDimensioner`. Color and cloud must cover the
    same field of view; their resolutions may differ. Boxes are measured against
    the active support plane, which can come from any source. Requires open3d,
    installed with ``pip install "depthai-nodes[box-measurement]"``.

    Attributes:
        out: Output stream of `depthai_nodes.message.BoxMeasurements`, one result
            per box detection with dimensions aggregated over the track.
    """

    def __init__(self) -> None:
        """Create the detection and dimensioning subnodes."""
        super().__init__()
        self._detector = self.createSubnode(BoxDetector)
        self._dimensioner = self.createSubnode(BoxDimensioner)

    def build(
        self,
        color: dai.Node.Output,
        pointCloud: dai.Node.Output,
        pointCloudSize: tuple[int, int],
        supportPlane: SupportPlane | None = None,
        confidenceThreshold: float = 0.5,
    ) -> "BoxMeasurer":
        """Build the detector and the dimensioner.

        Args:
            color: Color stream to detect boxes in.
            pointCloud: Output emitting organized ``dai.PointCloudData`` in
                millimeters.
            pointCloudSize: Cloud ``(width, height)`` in points.
            supportPlane: Initial active support plane; ``None`` leaves boxes
                ineligible until `setSupportPlane` is called.
            confidenceThreshold: Minimum detection confidence, between zero and one.

        Returns:
            The configured node instance.

        Raises:
            ImportError: If open3d is not installed.
            RuntimeError: If the node is already built.
            TypeError: If ``pointCloudSize`` is not a pair of integers,
                ``supportPlane`` is not a ``SupportPlane`` or
                ``confidenceThreshold`` is not a real number.
            ValueError: If ``pointCloudSize`` does not have two positive values,
                if ``confidenceThreshold`` is not finite or not between zero and
                one, or if the device is not an RVC4.
        """
        # Check open3d and the cloud size before the detector fetches the model,
        # and build the dimensioner last so a detector failure leaves it unlinked
        # and the build can be retried.
        self._dimensioner._check_build(pointCloudSize, supportPlane)
        self._detector.build(color, confidenceThreshold)
        self._dimensioner.build(
            self._detector.out, pointCloud, pointCloudSize, supportPlane
        )
        return self

    @property
    def out(self) -> dai.Node.Output:
        """Return the stream of `depthai_nodes.message.BoxMeasurements`."""
        return self._dimensioner.out

    @property
    def boxCloudOutput(self) -> dai.Node.Output:
        """Valid points under all box masks, as ``dai.PointCloudData``.

        Returns:
            ``dai.Node.Output``: The box point-cloud output.
        """
        return self._dimensioner.boxCloudOutput

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
        """Change the minimum detection confidence while the pipeline runs.

        Args:
            confidenceThreshold: Minimum detection confidence, between zero and one.

        Raises:
            RuntimeError: If the node is not built yet.
            TypeError: If ``confidenceThreshold`` is not a real number.
            ValueError: If ``confidenceThreshold`` is not finite or not between
                zero and one.
        """
        self._detector.setConfidenceThreshold(confidenceThreshold)

    def run(self) -> None:
        """Do nothing: the subnodes do the work."""
        return
