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

    def set_token(self, token: str):
        os.environ.setdefault("DEPTHAI_HUB_API_KEY", token)

    def set_cache_dir(self, cache_dir):
        """Set the cache directory for storing cached data.

        By default, the cache directory is set to /internal/private
        """

        self._em.setCacheDir(cache_dir)
        logger.info(f"Set cache directory to: {cache_dir}")

    def set_cache_if_cannot_send(self, cache_if_cannot_upload):
        """Set whether to cache data if it cannot be sent.

        By default, cacheIfCannotSend is set to false
        """

        self._em.setCacheIfCannotSend(cache_if_cannot_upload)
        if cache_if_cannot_upload:
            logger.info("Enabled caching if snaps cannot be sent.")
        else:
            logger.info("Disabled caching if snaps cannot be sent.")

    def set_log_response(self, log_response):
        """Set whether to log the responses from the server.

        By default, logResponse is set to false. Logs are visible in depthAI logs with
        INFO level.
        """

        self._em.setLogResponse(log_response)
        if log_response:
            logger.info("Enabled logging of server responses.")
        else:
            logger.info("Disabled logging of server responses.")

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
