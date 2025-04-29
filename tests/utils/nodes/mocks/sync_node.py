from .input import InputMock
from .node import NodeMock


class AutoCreateDict(dict):
    """Dictionary that automatically creates entries when accessed."""

    def __getitem__(self, key):
        if key not in self:
            self[key] = InputMock()
        return super().__getitem__(key)


class SyncMock(NodeMock):
    def __init__(self):
        super().__init__()
        self.inputs = AutoCreateDict()

    def setRunOnHost(self, run_on_host: bool):
        self._run_on_host = run_on_host

    def runOnHost(self):
        return self._run_on_host
