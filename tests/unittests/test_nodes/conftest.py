import time
from collections import deque
from queue import PriorityQueue
from typing import List, Optional, Tuple, Type, Union
from depthai_nodes.message import DetectedRecognitions, ImgDetectionsExtended
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

class Input:
    """Input class to simulate the depthai input node."""

    def __init__(self):
        self._queue = Queue()

    def get(self):
        return self._queue.get()

    def send(self, message):
        self._queue.send(message)

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

    def sendProcessingToPipeline(self, send: bool):
        self._sendProcessingToPipeline = send

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


class DetectionsRecognitionsSyncMock:
    """Mock class for the depthai ThreadedHostNode.

    The class is used to create a mock pipeline for testing purposes.
    """
    FPS_TOLERANCE_DIVISOR = 2.0
    INPUT_CHECKS_PER_FPS = 100
    def __init__(self):
        self._is_running = True
        self._camera_fps = 30
        self._unmatched_recognitions: list[dai.NNData] = []
        self._recognitions_by_detection_ts: dict[float, list[dai.NNData]] = {}
        self._detections: dict[float, dai.ImgDetections | dai.SpatialImgDetections | ImgDetectionsExtended] = {}
        self._ready_timestamps = PriorityQueue()

        self.input_recognitions = Input()
        self.input_detections = Input()
        self._output = Output()

    @property
    def output(self):
        return self._output

    @property
    def out(self):
        return self._output

    @out.setter
    def out(self, node: Output) -> None:
        self._output = node
    
    def set_camera_fps(self, fps: int) -> None:
        self._camera_fps = fps

    def isRunning(self):
        return self._is_running

    def setIsRunning(self, is_running: bool):
        self._is_running = is_running

    def build(self) -> "ThreadedHostNodeMock":
        """Mock implementation of build method."""
        return self

    def run(self) -> None:
        """Run method that processes inputs."""
        while self.isRunning():
            input_recognitions = self.input_recognitions.get() if not self.input_recognitions.is_empty() else None
            if input_recognitions:
                self._add_recognition(input_recognitions)
                self._send_ready_data()

            input_detections = self.input_detections.get() if not self.input_detections.is_empty() else None
            if input_detections:
                self._add_detection(input_detections)
                self._send_ready_data()
    
    def _timestamps_in_tolerance(self, timestamp1: float, timestamp2: float) -> bool:
        """Check if two timestamps are within tolerance."""
        difference = abs(timestamp1 - timestamp2)
        return difference < (1 / self._camera_fps / self.FPS_TOLERANCE_DIVISOR)
    
    def _get_total_seconds_ts(self, buffer_like: dai.Buffer) -> float:
        """Get timestamp in total seconds from a buffer."""
        return buffer_like.getTimestamp().total_seconds()
    
    def _add_recognition(self, recognition: dai.NNData) -> None:
        """Add a recognition to internal state."""
        recognition_ts = self._get_total_seconds_ts(recognition)
        best_matching_detection_ts = self._get_matching_detection_ts(recognition_ts)

        if best_matching_detection_ts is not None:
            self._add_recognition_by_detection_ts(
                recognition, best_matching_detection_ts
            )
            self._update_ready_timestamps(best_matching_detection_ts)
        else:
            self._unmatched_recognitions.append(recognition)
    
    def _get_matching_detection_ts(self, recognition_ts: float) -> float | None:
        """Find matching detection timestamp for a recognition."""
        for detection_ts in self._detections.keys():
            if self._timestamps_in_tolerance(detection_ts, recognition_ts):
                return detection_ts
        return None
    
    def _add_detection(
        self, detection: dai.ImgDetections | dai.SpatialImgDetections | ImgDetectionsExtended
    ) -> None:
        """Add a detection to internal state."""
        detection_ts = self._get_total_seconds_ts(detection)
        self._detections[detection_ts] = detection
        self._try_match_recognitions(detection_ts)
        self._update_ready_timestamps(detection_ts)
    
    def _try_match_recognitions(self, detection_ts: float) -> None:
        """Try to match unmatched recognitions with detection timestamp."""
        matched_recognitions: list[dai.NNData] = []
        for recognition in self._unmatched_recognitions:
            recognition_ts = self._get_total_seconds_ts(recognition)
            if self._timestamps_in_tolerance(detection_ts, recognition_ts):
                self._add_recognition_by_detection_ts(recognition, detection_ts)
                matched_recognitions.append(recognition)

        for matched_recognition in matched_recognitions:
            self._unmatched_recognitions.remove(matched_recognition)
    
    def _add_recognition_by_detection_ts(
        self, recognition: dai.NNData, detection_ts: float
    ) -> None:
        """Add recognition to mapping by detection timestamp."""
        if detection_ts in self._recognitions_by_detection_ts:
            self._recognitions_by_detection_ts[detection_ts].append(recognition)
        else:
            self._recognitions_by_detection_ts[detection_ts] = [recognition]
    
    def _update_ready_timestamps(self, timestamp: float) -> None:
        """Update ready timestamps queue."""
        if not self._timestamp_ready(timestamp):
            return

        self._ready_timestamps.put(timestamp)
    
    def _timestamp_ready(self, timestamp: float) -> bool:
        """Check if a timestamp is ready for processing."""
        detections = self._detections.get(timestamp)
        if not detections:
            return False
        elif len(detections.detections) == 0:
            return True

        recognitions = self._recognitions_by_detection_ts.get(timestamp)
        if not recognitions:
            return False

        return len(detections.detections) == len(recognitions)
    
    def _pop_ready_data(self) -> DetectedRecognitions | None:
        """Get next ready data and remove from internal state."""
        if self._ready_timestamps.empty():
            return None

        timestamp = self._ready_timestamps.get()
        detections = self._detections.pop(timestamp)
        recognitions = self._recognitions_by_detection_ts.pop(timestamp, None)

        return DetectedRecognitions(detections, recognitions)
    
    def _send_ready_data(self) -> None:
        """Send ready data to output queue."""
        ready_data = self._pop_ready_data()
        if ready_data:
            self._clear_old_data(ready_data)
            self.output.send(ready_data)
    
    def _clear_old_data(self, ready_data: DetectedRecognitions) -> None:
        """Clear data older than the given ready data."""
        current_timestamp = self._get_total_seconds_ts(ready_data)
        self._clear_unmatched_recognitions(current_timestamp)
        self._clear_old_detections(current_timestamp)
    
    def _clear_unmatched_recognitions(self, current_timestamp) -> None:
        """Clear unmatched recognitions older than timestamp."""
        unmatched_recognitions_to_remove = []
        for unmatched_recognition in self._unmatched_recognitions:
            if self._get_total_seconds_ts(unmatched_recognition) < current_timestamp:
                unmatched_recognitions_to_remove.append(unmatched_recognition)

        for unmatched_recognition in unmatched_recognitions_to_remove:
            self._unmatched_recognitions.remove(unmatched_recognition)
    
    def _clear_old_detections(self, current_timestamp) -> None:
        """Clear detections older than timestamp."""
        detection_keys_to_pop = []
        for detection_ts in self._detections.keys():
            if detection_ts < current_timestamp:
                detection_keys_to_pop.append(detection_ts)

        for detection_ts in detection_keys_to_pop:
            self._detections.pop(detection_ts)
            self._recognitions_by_detection_ts.pop(detection_ts, None)

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
        node = DetectionsRecognitionsSyncMock()
        node.parentPipeline = self
        self._nodes.append(node)
        return node


def pytest_configure():
    import depthai as dai

    dai.Pipeline = PipelineMock
    dai.node.ThreadedHostNode = ThreadedHostNodeMock
    dai.node.HostNode = HostNodeMock
