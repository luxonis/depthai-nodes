from typing import (
    Generic,
    TypeVar,
)

import depthai as dai

from depthai_nodes import Collection
from depthai_nodes.logging import get_logger

TCollected = TypeVar("TCollected", bound=dai.Buffer)


class MessageCollector(dai.node.ThreadedHostNode, Generic[TCollected]):
    """Batch timestamp-matched messages from a single input stream.

    A group contains messages within half a frame interval of its first timestamp.
    The first newer message outside that tolerance emits the preceding group;
    older out-of-window messages are discarded. Output metadata comes from the
    last message added to the group. There is no explicit final flush when the
    stream stops, so the final group needs a newer message to be emitted.

    Attributes:
        out: Stream of ``Collection`` messages containing the matched items.
    """

    def __init__(self) -> None:
        """Create the collection input and output ports."""
        super().__init__()
        self._camera_fps: int | None = None

        self._data_input = self.createInput()
        self._out = self.createOutput()

        self._next_reference = None

        self._logger = get_logger(__name__)
        self._logger.debug("MessageCollector initialized")

    @property
    def out(self) -> dai.Node.Output:
        """Return the gathered output stream."""
        return self._out

    def setCameraFps(self, fps: int) -> None:
        """Set the frame rate used to compute timestamp matching tolerance.

        Args:
            fps: Positive frame rate; the matching window is ``1 / (2 * fps)`` seconds.

        Raises:
            ValueError: If the frame rate is not positive.
        """
        if fps <= 0:
            raise ValueError(f"Camera FPS must be positive, got {fps}")
        self._camera_fps = fps
        self._logger.debug(f"Camera FPS set to {fps}")

    def build(
        self,
        cameraFps: int,
        inputData: dai.Node.Output,
    ) -> "MessageCollector[TCollected]":
        """Connect the data stream and set its timestamp tolerance.

        Args:
            cameraFps: Positive frame rate. Matching tolerance is half its frame
                interval.
            inputData: Input stream of messages to batch.

        Returns:
            This collector node.

        Raises:
            ValueError: If the frame rate is not positive.
        """
        self.setCameraFps(cameraFps)
        inputData.link(self._data_input)
        self._logger.debug(f"GatherData built with cameraFps={cameraFps}")
        return self

    def run(self) -> None:
        """Read the input stream and emit groups when newer timestamps arrive.

        Raises:
            ValueError: If the camera frame rate has not been configured.
        """
        self._logger.debug("MessageCollector run started")
        if not self._camera_fps:
            raise ValueError(
                "Camera FPS not set. Call build() before starting the pipeline."
            )
        msg: TCollected = self._data_input.get()  # noqa
        current_msg_ts = self._get_total_seconds_ts(msg)
        collected = [msg]
        last_collected_msg = msg
        while self.isRunning():
            msg: TCollected = self._data_input.get()  # noqa
            msg_ts = self._get_total_seconds_ts(msg)
            if self._timestamps_in_tolerance(msg_ts, current_msg_ts):
                collected.append(msg)
                last_collected_msg = msg
            else:
                # skip old data
                if msg_ts < current_msg_ts:
                    continue
                else:
                    output_msg = Collection(
                        items=collected,
                    )
                    output_msg.setTimestampDevice(
                        last_collected_msg.getTimestampDevice()
                    )
                    output_msg.setTimestamp(last_collected_msg.getTimestamp())
                    output_msg.setSequenceNum(last_collected_msg.getSequenceNum())
                    self.out.send(output_msg)
                    current_msg_ts = msg_ts
                    collected.clear()
                    collected.append(msg)
                    last_collected_msg = msg

    def _get_total_seconds_ts(self, buffer_like: dai.Buffer) -> float:
        return buffer_like.getTimestamp().total_seconds()

    def _timestamps_in_tolerance(self, timestamp1: float, timestamp2: float) -> bool:
        difference = abs(timestamp1 - timestamp2)
        return difference < (1 / self._camera_fps / 2)
