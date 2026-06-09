import depthai as dai


class SnapData(dai.Buffer):
    """DepthAI-compatible message for representing a single snap event.

    Attributes
    ----------
    snap_name : str
        Logical name of the snap.
    file_group : dai.FileGroup
        Object containing the snap image and associated data.
    tags : list[str]
        Optional list of tags to include.
    extras : dict[str, str]
        Additional metadata.
    """

    def __init__(
        self,
        snap_name: str,
        file_group: dai.FileGroup,
        tags: list[str] | None = None,
        extras: dict[str, str] | None = None,
    ):
        super().__init__()
        self.snap_name = snap_name
        self.file_group = file_group
        self.tags = tags or []
        self.extras = extras or {}
