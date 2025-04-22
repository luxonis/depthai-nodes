from typing import List, Optional, Tuple

import depthai as dai

from .input import InputMock
from .node import NodeMock
from .output import OutputMock
from .pipeline import PipelineMock


class HostNodeMock(NodeMock):
    """Mock class for the depthai HostNode.

    The class is used to create a mock pipeline for testing purposes.
    """

    def __init__(self):
        super().__init__()
        self._output = OutputMock()
        self._parent_pipeline = None
        self._input = InputMock()
        self._linked_args: Optional[Tuple[OutputMock, ...]] = None
        self._sendProcessingToPipeline = False
        self._pipeline = PipelineMock()

    def link_args(self, *args):
        for arg in args:
            assert isinstance(arg, OutputMock)
        self._linked_args = args

    @property
    def out(self):
        return self._output

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, output):
        self._output = output

    def createOutput(self, possibleDatatypes: List[Tuple[dai.DatatypeEnum, bool]]):
        return self._output

    def sendProcessingToPipeline(self, send: bool):
        self._sendProcessingToPipeline = send

    def getParentPipeline(self):
        return self._pipeline
