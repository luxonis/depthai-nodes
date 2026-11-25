"""Paint one view's box detections into a label image a PointCloud carries."""

import depthai as dai
import numpy as np

from depthai_nodes.message import BoxDetections
from depthai_nodes.node.box_associator.box_label import encode
from depthai_nodes.node.box_associator.host_messages import send_all, stamp_like


class BoxLabeler(dai.node.HostNode):
    """Paint each box's track and view into an RGB888i image at the depth's size.

    A box pixel is the ``box_label.encode`` colour of its view and track ID;
    background is black.

    Attributes:
        out: Output stream of RGB888i label ``ImgFrame``s.
        passthroughDetections: Output stream of the painted ``BoxDetections``, as a
            new message with the depth frame's timestamps and sequence number.
    """

    def __init__(self) -> None:
        """Initialize the labeler for view 0."""
        super().__init__()
        self.passthroughDetections = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )
        self._view = 0

    def build(
        self, detections: dai.Node.Output, depth: dai.Node.Output, view: int
    ) -> "BoxLabeler":
        """Link one view's detections and the depth aligned to its color.

        Args:
            detections: Output producing ``BoxDetections``.
            depth: Output producing depth frames aligned to the view's color.
            view: Index of the view, painted as view + 1 in the blue channel.

        Returns:
            This node.
        """
        self._view = view
        self.link_args(detections, depth)
        return self

    def process(self, detections: dai.Buffer, depth: dai.ImgFrame) -> None:
        """Paint and send the label image and passthrough detections of one frame.

        Args:
            detections: The view's ``BoxDetections``.
            depth: The depth frame paired with them.

        Raises:
            TypeError: If ``detections`` is not ``BoxDetections``.
        """
        if not isinstance(detections, BoxDetections):
            raise TypeError("Expected BoxDetections")
        image = dai.ImgFrame()
        image.setFrame(self._paint(detections, depth.getWidth(), depth.getHeight()))
        image.setType(dai.ImgFrame.Type.RGB888i)
        image.setWidth(depth.getWidth())
        image.setHeight(depth.getHeight())
        # Same geometry and stamps as the depth, so PointCloud pairs them exactly.
        image.setTransformation(depth.getTransformation())
        stamp_like(image, depth)
        # A new message: other consumers may hold the input, and
        # ``Collection.copy()`` returns a bare ``Collection``.
        passthrough = BoxDetections(
            list(detections.items), detections.segmentation_mask
        )
        stamp_like(passthrough, depth)
        send_all((self.out, image), (self.passthroughDetections, passthrough))

    def _paint(self, detections: BoxDetections, width: int, height: int) -> np.ndarray:
        """Return the ``(height, width, 3)`` label pixels of the detections."""
        palette = np.zeros((256, 3), np.uint8)
        for index, item in enumerate(detections.items):
            palette[index] = encode(self._view, item.track_id)
        # The mask covers the whole frame, stretched to its own resolution.
        mask = detections.segmentation_mask
        rows = np.arange(height) * mask.shape[0] // height
        cols = np.arange(width) * mask.shape[1] // width
        return palette[mask[rows[:, None], cols[None, :]]]
