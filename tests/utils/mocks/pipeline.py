from typing import List, Tuple, Type

import depthai as dai

from .device import DeviceMock
from .input import InfiniteInputMock, InputMock
from .output import OutputMock
from .threaded_host_node import ThreadedHostNodeMock


class PipelineMock:
    """Mock class for the depthai Pipeline.

    The class is used to create a mock pipeline for testing purposes.
    """

    def __init__(self):
        self._nodes = []
        self._defaultDevice = DeviceMock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def getDefaultDevice(self):
        return self._defaultDevice

    def remove(self, node):
        self._nodes.remove(node)

    def create(self, node_type: Type[ThreadedHostNodeMock]):
        from depthai_nodes.node.parsers.base_parser import BaseParser

        if issubclass(node_type, BaseParser):
            # Create a concrete subclass of the parser that implements the abstract methods
            class ParserMock(node_type):
                """Concrete parser class that implements the abstract methods of the
                BaseParser."""

                def __init__(self, parent_pipeline: PipelineMock):
                    self.parent_pipeline = parent_pipeline
                    super().__init__()
                    self._input = InfiniteInputMock()
                    self._out = OutputMock()
                    self._is_running = True

                @property
                def input(self):
                    return self._input

                @input.setter
                def input(self, input: InfiniteInputMock):
                    self._input = input

                @property
                def out(self):
                    return self._out

                @out.setter
                def out(self, output: OutputMock):
                    self._out = output

                def createInput(self):
                    self._input = InfiniteInputMock()
                    return self._input

                def createOutput(
                    self, possibleDatatypes: List[Tuple[dai.DatatypeEnum, bool]] = None
                ):
                    self._out = OutputMock()
                    return self._out

                def process(self, data: dai.NNData):
                    pass

                def isRunning(self):
                    return self._is_running

                def setIsRunning(self, is_running: bool):
                    self._is_running = is_running

                def getParentPipeline(self):
                    return self.parent_pipeline

            node = ParserMock(self)

        elif node_type == dai.node.DetectionParser:
            node = DetectionParserMock()

        else:

            class NodeMock(node_type):
                def __init__(self, pipeline):
                    self._pipeline = pipeline
                    super().__init__()
                    if (
                        self.__class__.__bases__[0].__bases__[0].__name__
                        == "ThreadedHostNodeMock"
                    ):
                        self._input = InfiniteInputMock()
                    else:
                        self._input = InputMock()
                    self._out = OutputMock()
                    self._is_running = True

                def getParentPipeline(self):
                    return self._pipeline

                @property
                def input(self):
                    return self._input

                @input.setter
                def input(self, input):
                    self._input = input

                @property
                def out(self):
                    return self._out

                @out.setter
                def out(self, output):
                    self._out = output

                def setNNArchive(self, nn_archive):
                    # check if superclass has the method
                    if hasattr(super(), "setNNArchive"):
                        super().setNNArchive(nn_archive)
                    else:
                        self._nn_archive = nn_archive

                def createInput(self):
                    if (
                        self.__class__.__bases__[0].__bases__[0].__name__
                        == "ThreadedHostNodeMock"
                    ):
                        self._input = InfiniteInputMock()
                    else:
                        self._input = InputMock()
                    return self._input

                def isRunning(self):
                    return self._is_running

                def setIsRunning(self, is_running: bool):
                    self._is_running = is_running

                def createOutput(
                    self, possibleDatatypes: List[Tuple[dai.DatatypeEnum, bool]] = None
                ):
                    self._out = OutputMock()
                    return self._out

            node = NodeMock(self)

        node.parentPipeline = self
        self._nodes.append(node)
        return node


class DetectionParserMock:
    def __init__(self):
        self._input = InputMock()
        self._out = OutputMock()
        self._nn_archive = None

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input):
        self._input = input

    @property
    def out(self):
        return self._out

    @out.setter
    def out(self, output):
        self._out = output

    def build(self):
        pass

    def setNNArchive(self, nn_archive):
        self._nn_archive = nn_archive
