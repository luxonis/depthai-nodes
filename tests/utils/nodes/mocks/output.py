import depthai as dai

from .queue import OutputQueueMock


class SourceNodeMock:
    """Node producing an output: on a device if given one, else on the host."""

    def __init__(self, device=None):
        self._device = device

    def runOnHost(self) -> bool:
        return self._device is None

    def getDevice(self):
        return self._device


class OutputMock:
    """Output class to simulate the depthai output node.

    Its parent runs on ``device``, or on the host when no device is given.
    """

    def __init__(self, device=None):
        self._parent = SourceNodeMock(device)
        self._datatypes: list[tuple[dai.DatatypeEnum, bool]] = []
        self._queues: list[OutputQueueMock] = []
        self.links: list = []

    def setPossibleDatatypes(self, datatypes: list[tuple[dai.DatatypeEnum, bool]]):
        self._datatypes = datatypes

    def getPossibleDatatypes(self) -> list[tuple[dai.DatatypeEnum, bool]]:
        return self._datatypes

    def send(self, message):
        for queue in self._queues:
            queue.send(message)

    def createOutputQueue(
        self, checking_function=None, model_slug=None, parser_name=None
    ) -> OutputQueueMock:
        queue = OutputQueueMock(
            checking_function=checking_function,
            model_slug=model_slug,
            parser_name=parser_name,
        )
        self._queues.append(queue)
        return queue

    def link(self, node):
        self.links.append(node)

    def getParent(self):
        return self._parent

    def getParentPipeline(self):
        return None
