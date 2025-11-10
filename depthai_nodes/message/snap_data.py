from typing import Dict, List, Optional

import depthai as dai


class SnapData(dai.Buffer):
    """DepthAI-compatible message for representing a single snap event.

    Attributes
    ----------
    snap_name : str
        Logical name of the snap.
    file_name : str
        File name for the snap image.
    frame : dai.ImgFrame
        Captured image frame associated with the snap.
    detections : Optional[dai.ImgDetections]
        Optional detection data.
    tags : List[str]
        Optional list of tags to include.
    extras : Dict[str, str]
        Additional metadata.
    """

    def __init__(
        self,
        snap_name: str,
        frame: dai.ImgFrame,
        file_name: str = "",
        detections: Optional[dai.ImgDetections] = None,
        tags: Optional[List[str]] = None,
        extras: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.snap_name = snap_name
        self.file_name = file_name
        self.frame = frame
        self.detections = detections
        self.tags = tags or []
        self.extras = extras or {}
