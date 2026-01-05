from typing import Dict, List, Optional

import depthai as dai


class SnapData(dai.Buffer):
    """DepthAI-compatible message for representing a single snap event.

    Attributes
    ----------
    snap_name : str
        Logical name of the snap.
    file_group : dai.FileGroup()
        File Group instance containing all image related data
    tags : List[str]
        Optional list of tags to include.
    extras : Dict[str, str]
        Additional metadata.
    """

    def __init__(
        self,
        snap_name: str,
        file_group: dai.FileGroup,
        tags: Optional[List[str]] = None,
        extras: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.snap_name = snap_name
        self.file_group = file_group
        self.tags = tags or []
        self.extras = extras or {}
