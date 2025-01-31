from typing import List, Optional, Tuple

import depthai as dai


class Queue:
    def __init__(self):
        self._messages = []

    def get(self):
        return self._messages.pop(0)

    def send(self, item):
        self._messages.append(item)


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


class HostNodeMock:
    def __init__(self):
        self._output = Output()
        self._linked_args: Optional[Tuple[Output, ...]] = None

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


def pytest_configure():
    import depthai as dai

    dai.node.HostNode = HostNodeMock
