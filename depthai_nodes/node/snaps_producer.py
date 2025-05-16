import time
from typing import Callable, Dict, List, Optional

import depthai as dai

from depthai_nodes.logging import get_logger
from depthai_nodes.node.base_host_node import BaseHostNode


class SnapsProducer(BaseHostNode):
    def __init__(
        self,
    ):
        super().__init__()
        self.last_update = time.time()

        self._em = dai.EventsManager()
        self._em.setLogResponse(True)
        self._time_interval = 60
        self._process_fn = None
        self._logger = get_logger(__name__)

    def setToken(self, token: str):
        self._em.setToken(token)

    def setUrl(self, url: str):
        self._em.setUrl(url)

    def setTimeInterval(self, time_interval: float):
        self._time_interval = time_interval

    def setProcessFn(self, process_fn: Callable):
        self._process_fn = process_fn

    def build(
        self,
        frame: dai.Node.Output,
        msg: dai.Node.Output = None,
        time_interval: float = 60.0,
        process_fn: Optional[Callable] = None,
    ) -> "SnapsProducer":
        if msg is not None:
            self.link_args(frame, msg)
        self.setTimeInterval(time_interval)
        if process_fn is not None:
            self.setProcessFn(process_fn)
        return self

    def process(self, frame: dai.Buffer, msg: dai.Buffer = None):
        assert isinstance(frame, dai.ImgFrame)
        if self._process_fn is None:
            self.sendSnap("frame", frame)
        if self._process_fn is not None:
            self._process_fn(self, frame, msg)

    def sendSnap(
        self,
        name: str,
        frame: dai.ImgFrame,
        tags: List[dai.EventData] = [],  # noqa: B006
        groups: List[str] = [],  # noqa: B006
        extra_data: Dict[str, str] = {},  # noqa: B006
    ):
        now = time.time()
        if now > self.last_update + self._time_interval:
            self.last_update = now
            self._em.sendSnap(name, frame, tags, groups, extra_data)
            self._logger.info(f"Sent snap with name `{name}`")


# Example usage:

# def filter_detections(producer, frame, msg):
#     # some filtering on msg possible here ...
#     extra_data = {"last_update":str(producer.last_update)}
#     producer.sendSnap(name="test123", frame=frame, extra_data=extra_data)

# snaps_producer = pipeline.create(SnapsProducer).build(
#     frame = nn_with_parser.passthrough,
#     msg = nn_with_parser.out,
#     process_fn=filter_detections,
#     time_interval=5
# )
