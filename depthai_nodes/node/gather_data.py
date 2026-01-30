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
    """Threaded host node that groups (“gathers”) multiple data messages around a single
    reference message, matched by timestamp.

    The node receives two input streams:

    - **reference_input**: reference messages (e.g., detections) that define a
      grouping key (timestamp) and determine how many data items should be
      gathered for that reference.
    - **data_input**: messages to be collected for the nearest reference timestamp
      within a tolerance derived from the camera FPS.

    For each reference timestamp, the node waits until the number of gathered
    data messages equals `wait_count_fn(reference)`. Once ready, it emits a
    :class:`depthai_nodes.GatheredData` message containing the reference message
    and the gathered list.

    The default `wait_count_fn` uses ``len(reference.detections)``, which works
    out-of-the-box for messages that expose a ``detections`` attribute (e.g.
    ``dai.ImgDetections`` and ``ImgDetectionsExtended``).

    Notes
    -----
    - Timestamp matching uses ``Buffer.getTimestamp().total_seconds()`` and a
      tolerance of ``1 / (camera_fps * FPS_TOLERANCE_DIVISOR)``.
    - If ``wait_count_fn(reference) == 0``, the node emits immediately for that
      reference (with an empty gathered list).
    - The node periodically polls inputs using ``tryGet()`` at a rate derived
      from ``camera_fps`` and ``INPUT_CHECKS_PER_FPS``.

    Inputs
    ------
    _data_input : dai.Node.Input
        Stream of data messages to be gathered (type ``TGathered``).
    _reference_input : dai.Node.Input
        Stream of reference messages used for grouping and deciding how many
        items to gather (type ``TReference``).

    Outputs
    -------
    out : dai.Node.Output
        Emits :class:`depthai_nodes.GatheredData` objects with:
        ``reference_data`` (the matched reference) and ``gathered`` (list of data).

    Class Attributes
    ---------------
    FPS_TOLERANCE_DIVISOR : float
        Divides the per-frame time interval to compute timestamp matching tolerance.
        Higher values make matching stricter.
    INPUT_CHECKS_PER_FPS : int
        Number of polling iterations per frame interval. Effective loop sleep is
        ``1 / (INPUT_CHECKS_PER_FPS * camera_fps)``.
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

        self._data_input = self.createInput()
        self._reference_input = self.createInput()
        self._out = self.createOutput()

        self._logger = get_logger(__name__)
        self._logger.debug("GatherData initialized")

    @property
    def out(self) -> dai.Node.Output:
        return self._out

    def setCameraFps(self, fps: int) -> None:
        if fps <= 0:
            raise ValueError(f"Camera FPS must be positive, got {fps}")
        self._camera_fps = fps
        self._logger.debug(f"Camera FPS set to {fps}")

    def setWaitCountFn(self, fn: Callable[[TReference], int]) -> None:
        self._wait_count_fn = fn

    @staticmethod
    def _default_wait_count_fn(reference: TReference) -> int:
        assert isinstance(reference, HasDetections)
        return len(reference.detections)

    def build(
        self,
        camera_fps: int,
        input_data: dai.Node.Output,
        input_reference: dai.Node.Output,
        wait_count_fn: Optional[Callable[[TReference], int]] = None,
    ) -> "GatherData[TReference, TGathered]":
        """Configure the node and link pipeline outputs to this node's inputs.

        This method must be called before the pipeline is started.

        Parameters
        ----------
        camera_fps : int
            Camera frame rate used to derive timestamp matching tolerance and
            polling interval. Must be positive.
        input_data : dai.Node.Output
            Upstream output producing the data messages to gather.
        input_reference : dai.Node.Output
            Upstream output producing the reference messages.
        wait_count_fn : Callable[[TReference], int], optional
            Function that returns how many data messages are expected for a given
            reference message. If not provided, defaults to ``len(reference.detections)``.
            Returning 0 causes immediate emission for that reference.

        Returns
        -------
        GatherData[TReference, TGathered]
            The configured node instance (for chaining).

        Raises
        ------
        ValueError
            If ``camera_fps`` is not positive.

        Examples
        --------
        Use default behavior (wait for number of detections in the reference):

        >>> gather = pipeline.create(GatherData).build(
        ...     camera_fps=30,
        ...     input_data=some_data_out,
        ...     input_reference=detections_out,
        ... )

        Custom wait count:

        >>> def wait_two(_ref): return 2
        >>> gather = GatherData().build(
        ...     camera_fps=60,
        ...     input_data=some_data_out,
        ...     input_reference=detections_out,
        ...     wait_count_fn=wait_two,
        ... )
        """
        self.setCameraFps(camera_fps)
        if wait_count_fn is None:
            wait_count_fn = self._default_wait_count_fn
        self.setWaitCountFn(wait_count_fn)

        input_data.link(self._data_input)
        input_reference.link(self._reference_input)
        self._logger.debug("Linked input_data and input_reference to GatherData inputs")

        self._logger.debug(f"GatherData built with camera_fps={camera_fps}")
        return self

    def run(self) -> None:
        self._logger.debug("GatherData run started")
        if not self._camera_fps:
            raise ValueError(
                "Camera FPS not set. Call build() before starting the pipeine."
            )

        while self.isRunning():
            try:
                data: TGathered = self._data_input.tryGet()  # noqa
                reference: TReference = self._reference_input.tryGet()  # noqa
            except dai.MessageQueue.QueueException as e:
                self._logger.error(
                    f"GatherData failed to read data from queues. Exception: {e}"
                )
                break
            if data:
                self._logger.debug("Data input received")
                self._add_data(data)
                self._send_ready_data()
            if reference:
                self._logger.debug("Reference input received")
                self._add_reference(reference)
                self._send_ready_data()

            time.sleep(1 / self.INPUT_CHECKS_PER_FPS / self._camera_fps)

    def _add_data(self, data: TGathered) -> None:
        data_ts = self._get_total_seconds_ts(data)
        best_matching_reference_ts = self._get_matching_reference_ts(data_ts)

        if best_matching_reference_ts is not None:
            self._add_data_by_reference_ts(data, best_matching_reference_ts)
            self._update_ready_timestamps(best_matching_reference_ts)
        else:
            self._unmatched_data.append(data)

    def _add_reference(
        self,
        reference: TReference,
    ) -> None:
        reference_ts = self._get_total_seconds_ts(reference)
        self._reference_data[reference_ts] = reference
        self._try_match_data(reference_ts)
        self._update_ready_timestamps(reference_ts)

    def _send_ready_data(self) -> None:
        ready_data = self._pop_ready_data()
        if ready_data:
            self._clear_old_data(ready_data)
            self.out.send(ready_data)
            self._logger.debug("Gathered data sent")

    def _get_total_seconds_ts(self, buffer_like: dai.Buffer) -> float:
        return buffer_like.getTimestamp().total_seconds()

    def _get_matching_reference_ts(self, data_ts: float) -> Optional[float]:
        for reference_ts in self._reference_data.keys():
            if self._timestamps_in_tolerance(reference_ts, data_ts):
                return reference_ts
        return None

    def _add_data_by_reference_ts(self, data: TGathered, reference_ts: float) -> None:
        if reference_ts in self._data_by_reference_ts:
            self._data_by_reference_ts[reference_ts].append(data)
        else:
            self._data_by_reference_ts[reference_ts] = [data]

    def _update_ready_timestamps(self, timestamp: float) -> None:
        if not self._timestamp_ready(timestamp):
            return
        self._ready_timestamps.put(timestamp)

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

    def _clear_old_references(self, current_timestamp: float) -> None:
        reference_keys_to_pop = []
        for reference_ts in self._reference_data.keys():
            if reference_ts < current_timestamp:
                reference_keys_to_pop.append(reference_ts)

        for reference_ts in reference_keys_to_pop:
            self._reference_data.pop(reference_ts)
            self._data_by_reference_ts.pop(reference_ts, None)
