from typing import List, Optional, Tuple

import depthai as dai


class Queue:
    def __init__(self):
        self._messages = []

    def get(self):
        return self._messages.pop(0)

    def send(self, item):
        self._messages.append(item)

    def is_empty(self):
        return len(self._messages) == 0


class Output:
    def __init__(self):
        self._datatypes: List[Tuple[dai.DatatypeEnum, bool]] = []
        self._queues: List[Queue] = []

    def setPossibleDatatypes(self, datatypes: List[Tuple[dai.DatatypeEnum, bool]]):
        self._datatypes = datatypes

    def getPossibleDatatypes(self) -> List[Tuple[dai.DatatypeEnum, bool]]:
        return self._datatypes

    def send(self, message):
        for queue in self._queues:
            queue.send(message)

    def createOutputQueue(self):
        queue = Queue()
        self._queues.append(queue)
        return queue

    def getParentPipeline(self):
        return None


class Pipeline:
    def __init__(self):
        self._default_device = Device()

    def getDefaultDevice(self):
        return self._default_device


class Device:
    def __init__(self):
        self._platform = dai.Platform(0)  # initialize platform with 0 (RVC2)

    def getPlatformAsString(self):
        return self._platform.name


class HostNodeMock:
    def __init__(self):
        self._output = Output()
        self._linked_args: Optional[Tuple[Output, ...]] = None
        self._pipeline = Pipeline()

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

    def sendProcessingToPipeline(self, send: bool):
        self._sendProcessingToPipeline = send

    def getParentPipeline(self):
        return self._pipeline


def pytest_configure():
    import depthai as dai

    dai.node.HostNode = HostNodeMock
