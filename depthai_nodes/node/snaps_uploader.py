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
        os.environ["DEPTHAI_HUB_API_KEY"] = token

    def setCacheDir(self, cacheDir: str):
        """Set the cache directory for storing cached data.

        By default, the cache directory is set to /internal/private
        """

        self._em.setCacheDir(cacheDir)
        logger.info(f"Set cache directory to: {cacheDir}")

    def setCacheIfCannotSend(self, cacheIfCannotUpload: bool):
        """Set whether to cache data if it cannot be sent.

        By default, cacheIfCannotSend is set to false
        """

        self._em.setCacheIfCannotSend(cacheIfCannotUpload)
        logger.info(f"Cache snaps if they cannot be uploaded: {cacheIfCannotUpload}")

    def setLogResponse(self, logResponse: bool):
        """Set whether to log the responses from the server.

        By default, logResponse is set to false. Logs are visible in depthAI logs with
        INFO level.
        """

        self._em.setLogResponse(logResponse)
        logger.info(f"Log server responses: {logResponse}")

    def build(self, snaps: dai.Node.Output):
        self.link_args(snaps)
        return self

    def process(self, snap: dai.Buffer):
        assert isinstance(snap, SnapData), f"Expected SnapData, got {type(snap)}"

        logger.debug(f"Sending snap: {snap.snap_name}")
        success = self._em.sendSnap(
            name=snap.snap_name,
            fileGroup=snap.file_group,
            tags=snap.tags,
            extras=snap.extras,
        )
        if success:
            logger.info(f"Snap '{snap.snap_name}' sent successfully.")
        else:
            logger.error(f"Failed to send snap '{snap.snap_name}'.")
