"""Association of tracked box detections from one or more views into box clouds."""

from collections.abc import Sequence

import depthai as dai

from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.box_associator.box_cloud_assembler import (
    CLOUD_KEY,
    BoxCloudAssembler,
    detections_key,
)
from depthai_nodes.node.box_associator.box_labeler import BoxLabeler

# A label colour (`box_label`) carries view + 1 in one byte, 0 being background.
MAX_VIEWS = 255


class BoxAssociator(BaseThreadedHostNode):
    """Associate tracked box detections from one or more views into box clouds.

    Emits one `depthai_nodes.message.BoxClouds` per frame on ``out``. Each view is
    a `BoxDetector` output plus depth aligned to that view's color stream, and is
    identified by its index in the views given to `build`. It takes at most 255
    views.

    Points are in millimetres, in shared coordinates. With several devices, the app
    sets the multi-device calibration on the pipeline; with one view, shared
    coordinates are the frame of the depth's extrinsics, which is the color camera
    for depth aligned to color.

    With several devices, their capture must be frame-synced (PTP). A frame that
    lacks any view's frame produces no output. If any view's device is lost or any
    view's stream stalls, output stops for good while the pipeline keeps running;
    an app that wants an error calls ``pipeline.setStopOnDeviceLoss(True)``.

    Requires open3d, installed with ``pip install "depthai-nodes[box-measurement]"``.
    """

    def __init__(self) -> None:
        """Create the node; `build` creates the subnodes."""
        super().__init__()
        self._assembler: BoxCloudAssembler | None = None

    def build(
        self, views: Sequence[tuple[dai.Node.Output, dai.Node.Output]]
    ) -> "BoxAssociator":
        """Set the views whose box detections are associated.

        Args:
            views: One ``(detections, depth)`` pair per view. ``detections`` emits
                ``BoxDetections``. ``depth`` emits depth ``ImgFrame``s aligned to
                the view's color stream: they cover the same field of view as the
                frame the detections were made on, at any resolution. Depth that
                doesn't cover that field of view isn't detected, and boxes are
                then labelled at the wrong points.

        Returns:
            The configured node instance.

        Raises:
            ImportError: If open3d is not installed.
            RuntimeError: If the node is already built.
            ValueError: If ``views`` is empty or holds more than 255 views. Views
                that share an output are not checked.
        """
        try:
            import open3d  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__} requires open3d. Install it with "
                '`pip install "depthai-nodes[box-measurement]"`.'
            ) from e
        if self._assembler is not None:
            raise RuntimeError("BoxAssociator is already built")
        if not 1 <= len(views) <= MAX_VIEWS:
            raise ValueError(
                f"views must hold 1 to {MAX_VIEWS} views, got {len(views)}"
            )

        cloud = self.createSubnode(dai.node.PointCloud)
        cloud.initialConfig.setOrganized(True)
        cloud.sync.setRunOnHost(True)
        pairing = self.createSubnode(dai.node.Sync)
        pairing.setRunOnHost(True)
        for view, (detections, depth) in enumerate(views):
            labeler = self.createSubnode(BoxLabeler).build(detections, depth, view)
            depth.link(cloud.getDepthInput(str(view)))
            labeler.out.link(cloud.getColorInput(str(view)))
            labeler.passthroughDetections.link(pairing.inputs[detections_key(view)])
        cloud.outputPointCloud.link(pairing.inputs[CLOUD_KEY])
        self._assembler = self.createSubnode(BoxCloudAssembler).build(pairing.out)
        return self

    @property
    def out(self) -> dai.Node.Output:
        """Return the stream of ``BoxClouds``, one per frame every view arrived in.

        Raises:
            RuntimeError: If the node is not built yet.
        """
        if self._assembler is None:
            raise RuntimeError("BoxAssociator is not built yet")
        return self._assembler.out

    def run(self) -> None:
        """Do nothing: the subnodes do the work."""
        return
