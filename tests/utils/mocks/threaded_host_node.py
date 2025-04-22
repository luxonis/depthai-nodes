from .input import InfiniteInputMock
from .node import NodeMock
from .queue import OutputQueueMock


class ThreadedHostNodeMock(NodeMock):
    """Mock class for the depthai ThreadedHostNode.

    The class is used to create a mock pipeline for testing purposes.
    """

    def __init__(self):
        super().__init__()
        self._output = OutputQueueMock()
        self._parent_pipeline = None
        self._input = InfiniteInputMock()

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @property
    def output(self):
        return self._output

    @property
    def out(self):
        return self._output

    @out.setter
    def out(self, node: OutputQueueMock) -> None:
        self._output = node

    @property
    def parentPipeline(self):
        return self._parent_pipeline

    @parentPipeline.setter
    def parentPipeline(self, pipeline):
        self._parent_pipeline = pipeline

    def getParentPipeline(self):
        return self._parent_pipeline
