from .input import InputMock
from .node import NodeMock
from .output import OutputMock
from .sync_node import SyncMock


class PointCloudConfigMock:
    """Stand-in for ``dai.PointCloudConfig``: only whether the cloud is organized."""

    def __init__(self):
        self._organized = False

    def setOrganized(self, organized: bool):
        self._organized = organized

    def getOrganized(self) -> bool:
        return self._organized


class PointCloudMock(NodeMock):
    """Mock of ``dai.node.PointCloud`` with named depth and color inputs.

    Each named input is created on first use and kept, so a test can look it up by name.
    """

    def __init__(self):
        super().__init__()
        self.initialConfig = PointCloudConfigMock()
        self.outputPointCloud = OutputMock()
        self.sync = SyncMock()
        self._depth_inputs: dict[str, InputMock] = {}
        self._color_inputs: dict[str, InputMock] = {}

    def getDepthInput(self, name: str) -> InputMock:
        return self._depth_inputs.setdefault(name, InputMock())

    def getColorInput(self, name: str) -> InputMock:
        return self._color_inputs.setdefault(name, InputMock())
