from typing import List, Tuple

import depthai as dai

from .queue import OutputQueueMock


class OutputMock:
    """Output class to simulate the depthai output node."""

    def __init__(self):
        self._datatypes: List[Tuple[dai.DatatypeEnum, bool]] = []
        self._queues: List[OutputQueueMock] = []

    def setPossibleDatatypes(self, datatypes: List[Tuple[dai.DatatypeEnum, bool]]):
        self._datatypes = datatypes

    def getPossibleDatatypes(self) -> List[Tuple[dai.DatatypeEnum, bool]]:
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
        pass

    def getParentPipeline(self):
        return None
