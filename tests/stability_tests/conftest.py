import time
from collections import deque
from queue import PriorityQueue
from typing import List, Optional, Tuple, Type, Union

import depthai as dai
from pytest import Config

from depthai_nodes.message.detected_recognitions import DetectedRecognitions
from depthai_nodes.node.detections_recognitions_sync import DetectionsRecognitionsSync


class Queue:
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

    def send(self, item):
        super().send(item)

    def get(self):
        if time.time() - self.start_time > self.duration:
            raise dai.MessageQueue.QueueException
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

    def tryGet(self):
        return self._queue.tryGet()

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


class DetectionsRecognitionsSyncMock(ThreadedHostNodeMock):
    FPS_TOLERANCE_DIVISOR = 2.0
    INPUT_CHECKS_PER_FPS = 100

    def __init__(self):
        super().__init__()
        self._camera_fps = 30
        self._unmatched_recognitions = []
        self._recognitions_by_detection_ts = {}
        self._detections = {}
        self._ready_timestamps = PriorityQueue()

        # Create inputs and outputs
        self._input_recognitions = Input()
        self._input_detections = Input()

    @property
    def input_recognitions(self):
        return self._input_recognitions

    @input_recognitions.setter
    def input_recognitions(self, value):
        self._input_recognitions = value

    @property
    def input_detections(self):
        return self._input_detections

    @input_detections.setter
    def input_detections(self, value):
        self._input_detections = value

    def build(self):
        return self

    def set_camera_fps(self, fps: int):
        self._camera_fps = fps

    def isRunning(self):
        return True

    def run(self):
        """Mock implementation of the run method."""
        # Process all available recognition messages
        while True:
            input_recognitions = self.input_recognitions.tryGet()
            if input_recognitions is None:
                break

            self._add_recognition(input_recognitions)
            self._send_ready_data()

        # Process all available detection messages
        while True:
            input_detections = self.input_detections.tryGet()
            if input_detections is None:
                break
            self._add_detection(input_detections)
            self._send_ready_data()

    def _timestamps_in_tolerance(self, timestamp1, timestamp2):
        difference = abs(timestamp1 - timestamp2)
        return difference < (1 / self._camera_fps / self.FPS_TOLERANCE_DIVISOR)

    def _get_total_seconds_ts(self, buffer_like):
        return buffer_like.getTimestamp().total_seconds()

    def _add_recognition(self, recognition):
        recognition_ts = self._get_total_seconds_ts(recognition)
        best_matching_detection_ts = self._get_matching_detection_ts(recognition_ts)
        if best_matching_detection_ts is not None:
            self._add_recognition_by_detection_ts(recognition, best_matching_detection_ts)
            self._update_ready_timestamps(best_matching_detection_ts)
        else:
            self._unmatched_recognitions.append(recognition)

    def _get_matching_detection_ts(self, recognition_ts):
        for detection_ts in self._detections.keys():
            if self._timestamps_in_tolerance(detection_ts, recognition_ts):
                return detection_ts
        return None

    def _add_detection(self, detection):
        detection_ts = self._get_total_seconds_ts(detection)
        self._detections[detection_ts] = detection
        self._try_match_recognitions(detection_ts)
        self._update_ready_timestamps(detection_ts)

    def _try_match_recognitions(self, detection_ts):
        matched_recognitions = []
        for recognition in self._unmatched_recognitions:
            recognition_ts = self._get_total_seconds_ts(recognition)
            if self._timestamps_in_tolerance(detection_ts, recognition_ts):
                self._add_recognition_by_detection_ts(recognition, detection_ts)
                matched_recognitions.append(recognition)

        for matched_recognition in matched_recognitions:
            self._unmatched_recognitions.remove(matched_recognition)

    def _add_recognition_by_detection_ts(self, recognition, detection_ts):
        if detection_ts in self._recognitions_by_detection_ts:
            self._recognitions_by_detection_ts[detection_ts].append(recognition)
        else:
            self._recognitions_by_detection_ts[detection_ts] = [recognition]

    def _update_ready_timestamps(self, timestamp):
        if not self._timestamp_ready(timestamp):
            return
        self._ready_timestamps.put(timestamp)

    def _timestamp_ready(self, timestamp):
        detections = self._detections.get(timestamp)
        if not detections:
            return False
        elif len(detections.detections) == 0:
            return True

        recognitions = self._recognitions_by_detection_ts.get(timestamp)
        if not recognitions:
            return False

        return len(detections.detections) == len(recognitions)

    def _pop_ready_data(self):
        if self._ready_timestamps.empty():
            return None

        timestamp = self._ready_timestamps.get()
        detections_recognitions = DetectedRecognitions()
        detections_recognitions.img_detections = self._detections.pop(timestamp)
        detections_recognitions.nn_data = self._recognitions_by_detection_ts.pop(timestamp, None)
        return detections_recognitions

    def _send_ready_data(self):
        """Send ready data."""
        ready_data = self._pop_ready_data()
        if ready_data:
            self._clear_old_data(ready_data)
            self.out.send(ready_data)

    def _clear_old_data(self, ready_data):
        """Clear old data."""
        current_timestamp = self._get_total_seconds_ts(ready_data)
        self._clear_unmatched_recognitions(current_timestamp)
        self._clear_old_detections(current_timestamp)

    def _clear_unmatched_recognitions(self, current_timestamp):
        """Clear unmatched recognitions."""
        unmatched_recognitions_to_remove = []
        for unmatched_recognition in self._unmatched_recognitions:
            if self._get_total_seconds_ts(unmatched_recognition) < current_timestamp:
                unmatched_recognitions_to_remove.append(unmatched_recognition)

        for unmatched_recognition in unmatched_recognitions_to_remove:
            self._unmatched_recognitions.remove(unmatched_recognition)

    def _clear_old_detections(self, current_timestamp):
        """Clear old detections."""
        detection_keys_to_pop = []
        for detection_ts in self._detections.keys():
            if detection_ts < current_timestamp:
                detection_keys_to_pop.append(detection_ts)

        for detection_ts in detection_keys_to_pop:
            self._detections.pop(detection_ts)
            self._recognitions_by_detection_ts.pop(detection_ts, None)


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

        elif node_type == DetectionsRecognitionsSync:
            node = DetectionsRecognitionsSyncMock()

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
