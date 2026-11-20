import time
from collections.abc import Callable
from datetime import timedelta
from queue import PriorityQueue
from typing import (
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import depthai as dai

from depthai_nodes import GatheredData
from depthai_nodes.logging import get_logger


@runtime_checkable
class HasDetections(Protocol):
    """Protocol for references exposing a list of detections."""

    @property
    def detections(self) -> list:
        """Return the detections used to derive the default wait count."""
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
    data messages equals ``wait_count_fn(reference)``. Once ready, it emits a
    ``depthai_nodes.GatheredData`` message containing the reference message and the
    gathered items.

    The default ``wait_count_fn`` uses ``len(reference.detections)``, which works
    out-of-the-box for messages that expose a ``detections`` attribute (e.g.
    ``dai.ImgDetections``).

    ``preserveOrder`` holds later groups behind earlier references. Optional
    ``timeout`` and ``maxPendingFrames`` limits discard whole incomplete groups.
    These options require references to arrive in source timestamp order.
    Defaults retain ready-first, unbounded behavior.

    Attributes:
        FPS_TOLERANCE_DIVISOR (``float``): Divides the per-frame time interval to
            compute timestamp matching tolerance. Higher values make matching stricter.
        INPUT_CHECKS_PER_FPS (``int``): Number of polling iterations per frame interval.
            Effective loop sleep is ``1 / (INPUT_CHECKS_PER_FPS * camera_fps)``.

    Note:
        - Timestamp matching uses ``Buffer.getTimestamp().total_seconds()`` and a
          tolerance of ``1 / (camera_fps * FPS_TOLERANCE_DIVISOR)``.
        - If ``wait_count_fn(reference) == 0``, an empty group is ready immediately,
          but ordered output waits for earlier references.
        - The node periodically polls inputs using ``tryGet()`` at a rate derived
          from ``camera_fps`` and ``INPUT_CHECKS_PER_FPS``.

    Inputs:

    * ``_data_input : dai.Node.Input``: Stream of data messages to be gathered (type
      ``TGathered``).
    * ``_reference_input : dai.Node.Input``: Stream of reference messages used for
      grouping and deciding how many
      items to gather (type ``TReference``).

    Outputs:

    * ``out : dai.Node.Output``: Emits ``depthai_nodes.GatheredData`` objects with:
      ``reference_data`` (the matched reference) and ``items`` (list of data).
    """

    FPS_TOLERANCE_DIVISOR = 2.0
    INPUT_CHECKS_PER_FPS = 100

    def __init__(self) -> None:
        """Initializes the GatherData node."""
        super().__init__()
        self._camera_fps: int | None = None
        self._unmatched_data: list[TGathered] = []
        self._data_by_reference_ts: dict[float, list[TGathered]] = {}
        self._reference_data: dict[float, TReference] = {}
        self._ready_timestamps = PriorityQueue()
        self._arrived_at: dict[float, float] = {}
        self._finished_timestamp = float("-inf")
        self._timeout_seconds: float | None = None
        self._max_pending_frames: int | None = None
        self._preserve_order = False
        self._manage_pending = False
        self._wait_count_fn = self._default_wait_count_fn

        self._data_input = self.createInput()
        self._reference_input = self.createInput()
        self._out = self.createOutput()

        self._logger = get_logger(__name__)
        self._logger.debug("GatherData initialized")

    @property
    def out(self) -> dai.Node.Output:
        """Return the gathered output stream."""
        return self._out

    def setCameraFps(self, fps: int) -> None:
        """Set the camera frame rate used for timestamp matching.

        Args:
            fps: Positive camera frame rate used for matching tolerance and polling.

        Raises:
            ValueError: If the frame rate is not positive.
        """
        if fps <= 0:
            raise ValueError(f"Camera FPS must be positive, got {fps}")
        self._camera_fps = fps
        self._logger.debug(f"Camera FPS set to {fps}")

    def setWaitCountFn(self, fn: Callable[[TReference], int]) -> None:
        """Set the expected gathered-item count for each reference.

        Args:
            fn: Callback accepting the reference message and returning the number of
                items to wait for. A count of zero makes an empty group ready.
        """
        self._wait_count_fn = fn

    def build(
        self,
        cameraFps: int,
        inputData: dai.Node.Output,
        inputReference: dai.Node.Output,
        waitCountFn: Callable[[TReference], int] | None = None,
        *,
        timeout: timedelta | None = None,
        maxPendingFrames: int | None = None,
        preserveOrder: bool = False,
    ) -> "GatherData[TReference, TGathered]":
        """Connect the data and reference streams used for gathering.

        Args:
            cameraFps: Camera frame rate used to derive timestamp matching tolerance and
                polling interval.
            inputData: Upstream output producing the data messages to gather.
            inputReference: Upstream output producing the reference messages.
            waitCountFn: Optional callback returning the number of data messages
                expected for a given reference. If omitted, defaults to
                len(reference.detections).
            timeout: Positive wait from reference arrival before dropping an incomplete
                group, including when inputs are idle. None disables expiry.
            maxPendingFrames: Positive limit on pending references and unmatched data
                messages. Oldest incomplete references and orphan messages are dropped
                when their respective limit is exceeded. None disables the limits.
            preserveOrder: Emit groups in source timestamp order, holding later groups
                until earlier references complete or are dropped. References must
                arrive in order. Complete groups are emitted even after timeout.

        Returns:
            The configured node instance.

        Raises:
            ValueError: If FPS, timeout or the pending-frame limit is not positive.
        """
        if timeout is not None and timeout <= timedelta(0):
            raise ValueError("Timeout must be positive")
        if maxPendingFrames is not None and maxPendingFrames <= 0:
            raise ValueError("Maximum pending frame count must be positive")
        self.setCameraFps(cameraFps)
        self._timeout_seconds = timeout.total_seconds() if timeout is not None else None
        self._max_pending_frames = maxPendingFrames
        self._preserve_order = preserveOrder
        self._manage_pending = (
            preserveOrder or timeout is not None or maxPendingFrames is not None
        )
        if waitCountFn is None:
            waitCountFn = self._default_wait_count_fn
        self.setWaitCountFn(waitCountFn)

        inputData.link(self._data_input)
        inputReference.link(self._reference_input)
        self._logger.debug("Linked inputData and inputReference to GatherData inputs")

        self._logger.debug(f"GatherData built with cameraFps={cameraFps}")
        return self

    def run(self) -> None:
        """Poll both inputs, match messages by timestamp, and emit ready groups."""
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
                if not self._manage_pending:
                    self._send_ready_data()
            if reference:
                self._logger.debug("Reference input received")
                self._add_reference(reference)
            if reference or self._manage_pending:
                self._send_ready_data()

            time.sleep(1 / self.INPUT_CHECKS_PER_FPS / self._camera_fps)

    @staticmethod
    def _default_wait_count_fn(reference: TReference) -> int:
        assert isinstance(reference, HasDetections)
        return len(reference.detections)

    def _add_data(self, data: TGathered) -> None:
        data_ts = self._get_total_seconds_ts(data)
        if self._manage_pending and data_ts <= self._finished_timestamp:
            return
        best_matching_reference_ts = self._get_matching_reference_ts(data_ts)

        if best_matching_reference_ts is not None:
            self._add_data_by_reference_ts(data, best_matching_reference_ts)
            self._update_ready_timestamps(best_matching_reference_ts)
        else:
            self._unmatched_data.append(data)
            if self._max_pending_frames is not None:
                del self._unmatched_data[: -self._max_pending_frames]

    def _add_reference(
        self,
        reference: TReference,
    ) -> None:
        reference_ts = self._get_total_seconds_ts(reference)
        if self._manage_pending:
            if reference_ts <= self._finished_timestamp:
                return
            self._arrived_at.setdefault(reference_ts, time.monotonic())
        self._reference_data[reference_ts] = reference
        self._try_match_data(reference_ts)
        self._update_ready_timestamps(reference_ts)

    def _send_ready_data(self) -> None:
        if self._manage_pending:
            now = time.monotonic()
            for timestamp in sorted(self._reference_data):
                if timestamp not in self._reference_data:
                    continue
                ready = self._timestamp_ready(timestamp)
                expired = (
                    self._timeout_seconds is not None
                    and now - self._arrived_at[timestamp] >= self._timeout_seconds
                )
                overflow = (
                    self._max_pending_frames is not None
                    and len(self._reference_data) > self._max_pending_frames
                )
                if not ready and not expired and not overflow:
                    if self._preserve_order:
                        break
                    continue
                reference = self._reference_data.pop(timestamp)
                items = self._data_by_reference_ts.pop(timestamp, [])
                self._arrived_at.pop(timestamp)
                self._finished_timestamp = timestamp
                self._clear_unmatched_data(timestamp)
                if ready:
                    if not self._preserve_order:
                        self._clear_old_references(timestamp)
                    self.out.send(GatheredData(reference_data=reference, items=items))
                else:
                    self._logger.warning(
                        "Dropping incomplete group for sequence %s",
                        reference.getSequenceNum(),
                    )
            return
        ready_data = self._pop_ready_data()
        if ready_data:
            self._clear_old_data(ready_data)
            self.out.send(ready_data)
            self._logger.debug("Gathered data sent")

    def _get_total_seconds_ts(self, buffer_like: dai.Buffer) -> float:
        return buffer_like.getTimestamp().total_seconds()

    def _get_matching_reference_ts(self, data_ts: float) -> float | None:
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
        if self._manage_pending or not self._timestamp_ready(timestamp):
            return
        self._ready_timestamps.put(timestamp)

    def _try_match_data(self, reference_ts: float) -> None:
        matched_data: list[TGathered] = []
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

    def _pop_ready_data(self) -> GatheredData | None:
        if self._ready_timestamps.empty():
            return None

        timestamp = self._ready_timestamps.get()
        return GatheredData(
            reference_data=self._reference_data.pop(timestamp),
            items=self._data_by_reference_ts.pop(timestamp, None) or [],
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
            self._arrived_at.pop(reference_ts, None)
