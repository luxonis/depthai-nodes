from .queue import InfiniteQueueMock, QueueMock


class InputMock:
    """Input class to simulate the depthai input node."""

    def __init__(self):
        self._queue = QueueMock()

    def get(self):
        return self._queue.get()

    def tryGet(self):
        return self._queue.tryGet()

    def send(self, message):
        self._queue.send(message)


class InfiniteInputMock(InputMock):
    """Special input class that uses InfiniteQueue to simulate a long running test."""

    def __init__(self):
        self._queue = InfiniteQueueMock()
