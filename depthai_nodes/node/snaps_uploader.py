import logging
import os

import depthai as dai

from depthai_nodes.message import SnapData
from depthai_nodes.node.base_host_node import BaseHostNode

logger = logging.getLogger(__name__)


class SnapsUploader(BaseHostNode):
    """Host node responsible for receiving SnapData messages and sending snaps to
    DepthAI Hub Events API."""

    def __init__(self):
        super().__init__()
        self._em = dai.EventsManager()

    def setToken(self, token: str):
        """Set the Hub API token used for snap uploads."""
        os.environ.setdefault("DEPTHAI_HUB_API_KEY", token)

    def setUrl(self, url: str):
        """Set the Hub Events API base URL used for snap uploads."""
        os.environ.setdefault("DEPTHAI_HUB_EVENTS_BASE_URL", url)

    def build(self, snaps: dai.Node.Output):
        """Connect the ``SnapData`` stream to the uploader node."""
        self.link_args(snaps)
        return self

    def process(self, snap: dai.Buffer):
        """Upload the incoming ``SnapData`` payload to the Hub Events API."""
        assert isinstance(snap, SnapData), f"Expected SnapData, got {type(snap)}"

        logger.debug(f"Sending snap: {snap.snap_name} -> {snap.file_name}")
        fileGroup = dai.FileGroup()
        if snap.detections:
            fileGroup.addImageDetectionsPair(
                snap.file_name, snap.frame, snap.detections
            )
        else:
            fileGroup.addFile(snap.file_name, snap.frame)
        success = self._em.sendSnap(
            name=snap.snap_name,
            fileGroup=fileGroup,
            tags=snap.tags,
            extras=snap.extras,
        )
        if success:
            logger.info(f"Snap '{snap.snap_name}' sent successfully.")
        else:
            logger.error(f"Failed to send snap '{snap.snap_name}'.")
