import logging
import time
from collections import deque

import depthai as dai

from tests.utils.constants import LOG_INTERVAL


class QueueMock:
    """Classic queue to imitate the depthai message queue."""

    def __init__(self):
        self._messages = deque()

    def get(self):
        if len(self._messages) == 0:
            return None
        return self._messages.pop()

    def tryGet(self):
        if len(self._messages) == 0:
            return None
        return self._messages.pop()

    def getAll(self):
        messages = list(self._messages)
        self._messages.clear()
        return messages

    def send(self, item):
        self._messages.appendleft(item)

    def __len__(self):
        return len(self._messages)

    def is_empty(self):
        return len(self._messages) == 0


class InfiniteQueueMock(QueueMock):
    """Queue used in InfiniteInput to simulate a long running test.

    We can set the duration for how long the queue will be active and in the end raise
    an Exception that is caught by the parser.
    """

    def __init__(self):
        super().__init__()
        self.duration = 1  # seconds
        self.start_time = time.time()
        self.log_interval = LOG_INTERVAL  # seconds
        self.time_after_last_log = time.time()
        self.log_counter = 1
        self.logger = logging.getLogger(__name__)

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
            self.logger.info(
                f"Test running... {elapsed:.1f}s elapsed, {remaining:.1f}s remaining"
            )
            self.time_after_last_log = current_time
            self.log_counter += 1
        element = self._messages.pop()
        self.send(element)
        return element

    def tryGet(self):
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

        if len(self._messages) == 0:
            return None
        element = self._messages.pop()
        self.send(element)
        return element


class OutputQueueMock(QueueMock):
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
