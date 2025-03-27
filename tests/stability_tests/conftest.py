import time
from collections import deque
from typing import List, Optional, Tuple, Type, Union

import depthai as dai
from pytest import Config


class Queue:
    """Classic queue to imitate the depthai message queue."""

    def __init__(self):
        self._messages = deque()

    def get(self):
        if len(self._messages) == 0:
            return None
        return self._messages.pop()

    def send(self, item):
        self._messages.append(item)

    def __len__(self):
        return len(self._messages)


class InfiniteQueue(Queue):
    """Queue used in InfiniteInput to simulate a long running test.

    We can set the duration for how long the queue will be active and in the end raise
    an Exception that is caught by the parser.
    """

    def __init__(self):
        super().__init__()
        self.duration = 5  # seconds
        self.start_time = time.time()
        self.log_interval = 2  # seconds
        self.time_after_last_log = time.time()
        self.log_counter = 1

    def send(self, item):
        super().send(item)

    def get(self):
        current_time = time.time()
        if current_time - self.start_time > self.duration:
            raise dai.MessageQueue.QueueException

        # Log progress periodically
        elapsed = current_time - self.time_after_last_log
        if elapsed > self.log_interval:
            elapsed = self.log_counter * self.log_interval
            remaining = self.duration - elapsed
            print(f"Test running... {elapsed:.1f}s elapsed, {remaining:.1f}s remaining")
            self.time_after_last_log = current_time
            self.log_counter += 1
        element = self._messages.pop()
        self.send(element)
        return element


class OutputQueue(Queue):
    """Output queue used for checking the output messages from parsers.

    The messages are verified on the fly so the queue doesnt fill up on long running
    tests.
    """

    def __init__(self, checking_function=None, model_slug=None, parser_name=None):
        super().__init__()
        self._checking_function = checking_function
        self._model_slug = model_slug
        self._parser_name = parser_name

    def send(self, item):
        super().send(item)

        # check msg
        if self._checking_function is not None:
            self._checking_function(item, self._model_slug, self._parser_name)
            self._messages.pop()


class Input:
    """Input class to simulate the depthai input node."""

    def __init__(self):
        self._queue = Queue()

    def get(self):
        return self._queue.get()

    def send(self, message):
        self._queue.send(message)


class InfiniteInput(Input):
    """Special input class that uses InfiniteQueue to simulate a long running test."""

    def __init__(self):
        self._queue = InfiniteQueue()


class Output:
    """Output class to simulate the depthai output node."""

    def __init__(self):
        self._datatypes: List[Tuple[dai.DatatypeEnum, bool]] = []
        self._queues: List[OutputQueue] = []

    def setPossibleDatatypes(self, datatypes: List[Tuple[dai.DatatypeEnum, bool]]):
        self._datatypes = datatypes

    def getPossibleDatatypes(self) -> List[Tuple[dai.DatatypeEnum, bool]]:
        return self._datatypes

    def send(self, message):
        for queue in self._queues:
            queue.send(message)

    def createOutputQueue(
        self, checking_function, model_slug, parser_name
    ) -> OutputQueue:
        queue = OutputQueue(
            checking_function=checking_function,
            model_slug=model_slug,
            parser_name=parser_name,
        )
        self._queues.append(queue)
        return queue

    def link(self, node):
        pass


class HostNodeMock:
    """Mock class for the depthai HostNode.

    The class is used to create a mock pipeline for testing purposes.
    """

    def __init__(self):
        self._output = OutputQueue()
        self._parent_pipeline = None
        self._input = Input()
        self._linked_args: Optional[Tuple[Output, ...]] = None
        self._sendProcessingToPipeline = False

    def link_args(self, *args):
        for arg in args:
            assert isinstance(arg, Output)
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


class ThreadedHostNodeMock:
    """Mock class for the depthai ThreadedHostNode.

    The class is used to create a mock pipeline for testing purposes.
    """

    def __init__(self):
        self._output = OutputQueue()
        self._parent_pipeline = None
        self._input = Input()

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
    def out(self, node: OutputQueue) -> None:
        self._output = node

    @property
    def parentPipeline(self):
        return self._parent_pipeline

    @parentPipeline.setter
    def parentPipeline(self, pipeline):
        self._parent_pipeline = pipeline

    def getParentPipeline(self):
        return self._parent_pipeline


class NeuralNetworkMock:
    def __init__(self):
        self._input = Input()
        self._out = Output()
        self._passthrough = Output()
        self._nn_archive = None

    def build(
        self,
        input: Input,
        model: Union[dai.NNModelDescription, dai.NNArchive],
        fps: float,
    ):
        self._nn_archive = model
        self._input = input
        return self


class DetectionParserMock:
    def __init__(self):
        self._input = Input()
        self._out = Output()
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


class DeviceMock:
    def __init__(self):
        pass

    def getPlatformAsString(self):
        return "RVC2"


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
                    self._input = InfiniteInput()
                    self._out = Output()
                    self._is_running = True

                @property
                def input(self):
                    return self._input

                @input.setter
                def input(self, input: InfiniteInput):
                    self._input = input

                @property
                def out(self):
                    return self._out

                @out.setter
                def out(self, output: Output):
                    self._out = output

                def createInput(self):
                    self._input = InfiniteInput()
                    return self._input

                def createOutput(
                    self, possibleDatatypes: List[Tuple[dai.DatatypeEnum, bool]] = None
                ):
                    self._out = Output()
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
                    self._input = Input()
                    self._out = Output()

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

            node = NodeMock(self)

        node.parentPipeline = self
        self._nodes.append(node)
        return node


def pytest_configure(config: Config):
    import depthai as dai

    dai.Pipeline = PipelineMock
    dai.node.ThreadedHostNode = ThreadedHostNodeMock
    dai.node.HostNode = HostNodeMock
