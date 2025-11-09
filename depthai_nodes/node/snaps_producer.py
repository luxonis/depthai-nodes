import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import depthai as dai

from depthai_nodes.node.base_host_node import BaseHostNode


@dataclass
class SnapData:
    """Represents a single snap event to be sent.

    Attributes
    ----------
    name : str
        Logical name of the snap.
    file_name : str
        File name for the snap image.
    frame : dai.ImgFrame
        Captured image frame.
    detections : Optional[dai.ImgDetections]
        Optional detection data associated with the frame.
    tags : List[str]
        Optional list of tags to include.
    extras : Dict[str, str]
        Optional extra metadata to attach.
    """

    name: str
    file_name: str
    frame: dai.ImgFrame
    detections: Optional[dai.ImgDetections] = None
    tags: List[str] = field(default_factory=list)
    extras: Dict[str, str] = field(default_factory=dict)


class SnapsProducer(BaseHostNode):
    """A host node responsible solely for sending snaps to DepthAI Hub Events API.

    Attributes
    ----------
    _em : dai.EventsManager
        Internal DepthAI EventsManager instance used for snap transmission.
    _running : bool
        Flag indicating whether snap sending is currently enabled.
    """

    def __init__(self):
        super().__init__()
        self._em = dai.EventsManager()

    def set_token(self, token: str):
        os.environ.setdefault("DEPTHAI_HUB_API_KEY", token)

    def set_url(self, url: str):
        os.environ.setdefault("DEPTHAI_HUB_EVENTS_BASE_URL", url)

    def build(self, snaps_buffer: dai.Node.Output):
        self.link_args(snaps_buffer)
        return self

    def process(self, snaps_buffer: dai.Buffer):
        snaps: list[SnapData] = pickle.loads(snaps_buffer.getData())

        for snap in snaps:
            self._em.sendSnap(
                name=snap.name,
                fileName=snap.file_name,
                imgFrame=snap.frame,
                imgDetections=snap.detections,
                tags=snap.tags,
                extras=snap.extras,
            )
