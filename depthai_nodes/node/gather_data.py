import time
from queue import PriorityQueue
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import depthai as dai

from depthai_nodes import GatheredData
from depthai_nodes.logging import get_logger


@runtime_checkable
class HasDetections(Protocol):
    @property
    def detections(self) -> List:
        ...


TReference = TypeVar("TReference", bound=dai.Buffer)
TGathered = TypeVar("TGathered", bound=dai.Buffer)


class GatherData(dai.node.ThreadedHostNode, Generic[TReference, TGathered]):
    """A class for gathering data. Gathers n messages based on reference_data. To
    determine n, wait_count_fn function is used. The default wait_count_fn function is
    waiting for len(TReference.detection). This means the node works out-of-the-box with
    dai.ImgDetections and ImgDetectionsExtended.

    Attributes
    ----------
    FPS_TOLERANCE_DIVISOR: float
        Divisor for the FPS tolerance.
    INPUT_CHECKS_PER_FPS: int
        Number of input checks per FPS.
    input_data: dai.Node.Input
        Input to be gathered.
    input_reference: dai.Node.Input
        Input to determine how many gathered items to wait for.
    output: dai.Node.Output
        Output for gathered data.
    """

    FPS_TOLERANCE_DIVISOR = 2.0
    INPUT_CHECKS_PER_FPS = 100

    def __init__(self) -> None:
        """Initializes the GatherData node."""
        super().__init__()
        self._camera_fps: Optional[int] = None
        self._unmatched_data: List[TGathered] = []
        self._data_by_reference_ts: Dict[float, List[TGathered]] = {}
        self._reference_data: Dict[float, TReference] = {}
        self._ready_timestamps = PriorityQueue()
        self._wait_count_fn = self._default_wait_count_fn

        self.input_data = self.createInput()
        self.input_reference = self.createInput()
        self.out = self.createOutput()

        self._logger = get_logger(__name__)
        self._logger.debug("GatherData initialized")

    @staticmethod
    def _default_wait_count_fn(reference: TReference) -> int:
        assert isinstance(reference, HasDetections)
        return len(reference.detections)

    def build(
        self,
        camera_fps: int,
        wait_count_fn: Optional[Callable[[TReference], int]] = None,
    ) -> "GatherData[TReference, TGathered]":
        """Builds and configures the GatherData node with the specified parameters.

        @param camera_fps: The frames per second (FPS) setting for the camera.
        @param wait_count_fn: A function that takes a reference and returns how many frames to wait.
        @type camera_fps: int
        @type wait_count_fn: Optional[Callable[[TReference], int]]

        @return: The configured GatherData node instance.
        @rtype: GatherData[TReference, TGathered]

        @example:
            >>> gather_node = GatherData()
            >>> gather_node.build(camera_fps=30)
            >>> def custom_wait(ref): return 2
            >>> gather_node.build(camera_fps=60, wait_count_fn=custom_wait)
        """
        self.set_camera_fps(camera_fps)
        if wait_count_fn is None:
            wait_count_fn = self._default_wait_count_fn
        self.set_wait_count_fn(wait_count_fn)
        self._logger.debug(f"GatherData built with camera_fps={camera_fps}")
        return self

    def set_camera_fps(self, fps: int) -> None:
        if fps <= 0:
            raise ValueError(f"Camera FPS must be positive, got {fps}")
        self._camera_fps = fps
        self._logger.debug(f"Camera FPS set to {fps}")

    def run(self) -> None:
        self._logger.debug("GatherData run started")
        if not self._camera_fps:
            raise ValueError("Camera FPS not set. Call build() before run().")

        while self.isRunning():
            try:
                input_data: TGathered = self.input_data.tryGet()
                input_reference: TReference = self.input_reference.tryGet()
            except dai.MessageQueue.QueueException:
                break
            if input_data:
                self._logger.debug("Input data received")
                self._add_data(input_data)
                self._send_ready_data()
            if input_reference:
                self._logger.debug("Input reference received")
                self._add_reference(input_reference)
                self._send_ready_data()

            time.sleep(1 / self.INPUT_CHECKS_PER_FPS / self._camera_fps)

    def _send_ready_data(self) -> None:
        ready_data = self._pop_ready_data()
        if ready_data:
            self._clear_old_data(ready_data)
            self.out.send(ready_data)
            self._logger.debug("Gathered data sent")

    def _add_data(self, data: TGathered) -> None:
        data_ts = self._get_total_seconds_ts(data)
        best_matching_reference_ts = self._get_matching_reference_ts(data_ts)

        if best_matching_reference_ts is not None:
            self._add_data_by_reference_ts(data, best_matching_reference_ts)
            self._update_ready_timestamps(best_matching_reference_ts)
        else:
            self._unmatched_data.append(data)

    def _get_matching_reference_ts(self, data_ts: float) -> Optional[float]:
        for reference_ts in self._reference_data.keys():
            if self._timestamps_in_tolerance(reference_ts, data_ts):
                return reference_ts
        return None

    def _add_reference(
        self,
        reference: TReference,
    ) -> None:
        reference_ts = self._get_total_seconds_ts(reference)
        self._reference_data[reference_ts] = reference
        self._try_match_data(reference_ts)
        self._update_ready_timestamps(reference_ts)

    def _try_match_data(self, reference_ts: float) -> None:
        matched_data: List[TGathered] = []
        for data in self._unmatched_data:
            data_ts = self._get_total_seconds_ts(data)
            if self._timestamps_in_tolerance(reference_ts, data_ts):
                self._add_data_by_reference_ts(data, reference_ts)
                matched_data.append(data)

        for matched in matched_data:
            self._unmatched_data.remove(matched)

    def _timestamps_in_tolerance(self, timestamp1: float, timestamp2: float) -> bool:
        difference = abs(timestamp1 - timestamp2)
        return difference < (1 / self._camera_fps / self.FPS_TOLERANCE_DIVISOR)

    def _add_data_by_reference_ts(self, data: TGathered, reference_ts: float) -> None:
        if reference_ts in self._data_by_reference_ts:
            self._data_by_reference_ts[reference_ts].append(data)
        else:
            self._data_by_reference_ts[reference_ts] = [data]

    def _update_ready_timestamps(self, timestamp: float) -> None:
        if not self._timestamp_ready(timestamp):
            return

        self._ready_timestamps.put(timestamp)

    def _timestamp_ready(self, timestamp: float) -> bool:
        reference = self._reference_data.get(timestamp)
        if not reference:
            return False

        wait_for_count = self._get_wait_count(reference)
        if wait_for_count == 0:
            return True

        recognitions = self._data_by_reference_ts.get(timestamp)
        if not recognitions:
            return False

        return wait_for_count == len(recognitions)

    def _get_wait_count(self, reference: TReference) -> int:
        return self._wait_count_fn(reference)

    def _pop_ready_data(self) -> Optional[GatheredData]:
        if self._ready_timestamps.empty():
            return None

        timestamp = self._ready_timestamps.get()
        return GatheredData(
            reference_data=self._reference_data.pop(timestamp),
            gathered=self._data_by_reference_ts.pop(timestamp, None) or [],
        )

    def _clear_old_data(self, ready_data: GatheredData) -> None:
        current_timestamp = self._get_total_seconds_ts(ready_data)
        self._clear_unmatched_data(current_timestamp)
        self._clear_old_references(current_timestamp)

    def _clear_unmatched_data(self, current_timestamp: float) -> None:
        unmatched_data_to_remove = []
        for unmatched_data in self._unmatched_data:
            if self._get_total_seconds_ts(unmatched_data) < current_timestamp:
                unmatched_data_to_remove.append(unmatched_data)

        for unmatched_data in unmatched_data_to_remove:
            self._unmatched_data.remove(unmatched_data)

    def _get_total_seconds_ts(self, buffer_like: dai.Buffer) -> float:
        return buffer_like.getTimestamp().total_seconds()

    def _clear_old_references(self, current_timestamp: float) -> None:
        reference_keys_to_pop = []
        for reference_ts in self._reference_data.keys():
            if reference_ts < current_timestamp:
                reference_keys_to_pop.append(reference_ts)

        for reference_ts in reference_keys_to_pop:
            self._reference_data.pop(reference_ts)
            self._data_by_reference_ts.pop(reference_ts, None)

    def set_wait_count_fn(self, fn: Callable[[TReference], int]) -> None:
        self._wait_count_fn = fn
