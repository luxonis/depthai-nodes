import logging
import os

import depthai as dai

from depthai_nodes.message import SnapData
from depthai_nodes.node.base_host_node import BaseHostNode

logger = logging.getLogger(__name__)


class SnapsUploader(BaseHostNode):
    """Upload ``SnapData`` messages through the native Hub EventsManager.

    Configure the API token and optional local caching before starting the pipeline.
    Upload failures are logged; offline caching is controlled by the native manager.
    """

    def __init__(self):
        super().__init__()
        self._em = dai.EventsManager()

    def setToken(self, token: str):
        """Set a Hub API token only when the environment does not already define one.

        Args:
            token: Value assigned to ``DEPTHAI_HUB_API_KEY`` if that environment
                variable is absent. An existing value is preserved.
        """
        os.environ.setdefault("DEPTHAI_HUB_API_KEY", token)

    def setCacheDir(self, cacheDir: str):
        """Set the EventsManager directory for cached uploads.

        Args:
            cacheDir: Local cache directory passed to the native EventsManager.
        """

        self._em.setCacheDir(cacheDir)
        logger.info(f"Set cache directory to: {cacheDir}")

    def setCacheIfCannotSend(self, cacheIfCannotUpload: bool):
        """Configure caching when a snap cannot be uploaded.

        Args:
            cacheIfCannotUpload: Whether the native EventsManager should cache unsent
                snaps.
        """

        self._em.setCacheIfCannotSend(cacheIfCannotUpload)
        logger.info(f"Cache snaps if they cannot be uploaded: {cacheIfCannotUpload}")

    def setLogResponse(self, logResponse: bool):
        """Configure native server-response logging.

        Args:
            logResponse: Whether upload responses appear in DepthAI INFO logs.
        """

        self._em.setLogResponse(logResponse)
        logger.info(f"Log server responses: {logResponse}")

    def build(self, snaps: dai.Node.Output):
        """Connect a stream of snap payloads.

        Args:
            snaps: Output producing ``SnapData`` messages.

        Returns:
            This uploader node.
        """
        self.link_args(snaps)
        return self

    def process(self, snap: dai.Buffer):
        """Submit one snap and log whether the native upload succeeded.

        Args:
            snap: ``SnapData`` containing the name, file group, tags, and metadata.

        Raises:
            AssertionError: If the input is not a ``SnapData`` message.
        """
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
