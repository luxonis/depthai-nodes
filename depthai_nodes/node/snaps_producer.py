import time
from typing import Callable, Dict, List, Optional, Union

import depthai as dai

from depthai_nodes.node.base_host_node import BaseHostNode

ProcessFnFrameOnlyType = Callable[["SnapsProducerFrameOnly", dai.ImgFrame], None]
ProcessFnType = Callable[["SnapsProducer", dai.ImgFrame, dai.Buffer], None]


class SnapsProducerFrameOnly(BaseHostNode):
    """A host node that helps with creating and sending snaps. If you also have
    additional message as input (e.g. detections) consider using `SnapsProducer` node
    instead.

    Attributes:
    ----------
    frame : dai.ImgFrame
        The input message for snap creation.
    time_interval : Union[int, float]
        Time interval between snaps if no custom process function is provided. Defaults to 60s.
    process_fn : Callable[['SnapsProducerFrameOnly', dai.ImgFrame], None]
        Custom processing function for business logic regarding snaps. If None defaults to sending only frames on time intervals.
    token : str
        Hub API token of the team you want to save snaps under. Can be also set through `DEPTHAI_HUB_API_KEY` env variable.
    url : str
        Custom URL for events service. By default, the URL is set to https://events-ingest.cloud.luxonis.com
    """

    def __init__(
        self,
        time_interval: Union[int, float] = 60.0,
        process_fn: Optional[ProcessFnFrameOnlyType] = None,
    ):
        super().__init__()
        self.last_update = time.time()

        self._em = dai.EventsManager()
        self._em.setLogResponse(True)
        self.setTimeInterval(time_interval)
        if process_fn is None:
            self._process_fn = None
        else:
            self.setProcessFn(process_fn)

        self._logger.debug(
            f"SnapsProducerFrameOnly initialized with time_interval={time_interval}"
        )

    def setToken(self, token: str):
        """Sets the Hub API token.

        @param token: Hub API token of your team.
        @type token: str
        """
        if not isinstance(token, str):
            raise ValueError("token must of of type string.")
        self._em.setToken(token)
        self._logger.debug(f"Token set to {token}")

    def setUrl(self, url: str):
        """Set the URL of the events service. By default, the URL is set to https://events-ingest.cloud.luxonis.com.

        @param url: URL of the events service.
        @type url: str
        """
        if not isinstance(url, str):
            raise ValueError("url must of of type string.")
        self._em.setUrl(url)
        self._logger.debug(f"Url set to {url}")

    def setTimeInterval(self, time_interval: Union[int, float]):
        """Sets time interval between snaps. Only relevant if using default processing.

        @param time_interval: Time interval between snaps for default sending in
            seconds.
        @type time_interval: Union[int, float]
        """
        if not (isinstance(time_interval, int) or isinstance(time_interval, float)):
            raise ValueError("time_interval must be of type int or float.")

        self.time_interval = time_interval
        self._logger.debug(f"Time interval set to {time_interval}")

    def setProcessFn(self, process_fn: ProcessFnFrameOnlyType):
        """Sets custom processing function.

        @param process_fn: Custom snaps processing function.
        @type process_fn: Callable[['SnapsProducerFrameOnly', dai.ImgFrame], None]
        """
        if not isinstance(process_fn, Callable):
            raise ValueError("process_fn must be a function.")

        self._process_fn = process_fn
        self._logger.debug("Process function set")

    def build(
        self,
        frame: dai.Node.Output,
        time_interval: Union[int, float] = 60.0,
        process_fn: Optional[ProcessFnFrameOnlyType] = None,
    ) -> "SnapsProducerFrameOnly":
        """Configures the node.

        @param frame: The input message for snap creation.
        @type frame: dai.Node.Output
        @param time_interval: Time interval between snaps for default sending in
            seconds. Defualts to 60.
        @type time_interval: Union[int, float]
        @param process_fn: Custom snaps processing function. Defaults to None.
        @type process_fn: Callable[['SnapsProducerFrameOnly', dai.ImgFrame], None]
        @return: The node object which handles snap creation and sending.
        @rtype: SnapsProducerFrameOnly
        """
        self.link_args(frame)
        self.setTimeInterval(time_interval)
        if process_fn is not None:
            self.setProcessFn(process_fn)
        self._logger.debug(
            f"SnapsProducerFrameOnly built with time_interval={time_interval}"
        )
        return self

    def process(self, frame: dai.Buffer):
        """Processes incoming frames and sends out snaps. If not custom process function
        is set then it sends only one frame every `time_interval` seconds. If using
        custom process function make sure to call `self.sendSnap()` at the end.

        @param frame: The input message for snap creation.
        @type frame: dai.ImgFrame
        """
        self._logger.debug("Processing new input")
        assert isinstance(frame, dai.ImgFrame)
        if self._process_fn is None:
            self.sendSnap("frame", frame)
        else:
            self._process_fn(self, frame)

    def sendSnap(
        self,
        name: str,
        frame: dai.ImgFrame,
        data: List[dai.EventData] = [],  # noqa: B006
        tags: List[str] = [],  # noqa: B006
        extra_data: Dict[str, str] = {},  # noqa: B006
        device_serial_num: str = "",
    ) -> bool:
        """Function that creates the snap and sends it out if time from last snap is
        greater than time_interval. Make sure to call this function from any custom
        processing functions to send snaps. Returns True if snap was sent.

        @param name: Name of the snap.
        @type name: str
        @param frame: Image frame to send.
        @type frame: dai.ImgFrame
        @param data: List of EventData objects to send. Defualts to [].
        @type data: List[dai.EventData]
        @param tags: List of tags to send. Defaults to [].
        @type tags: List[str]
        @param extra_data: Extra data to send. Defaults to {}.
        @type extra_data: Dict[str, str]
        @param device_serial_num: Device serial number. Defualts to ''.
        @type device_serial_num: str
        @return: True if snap was sent out else False.
        @rtype: bool
        """
        now = time.time()
        if now > self.last_update + self.time_interval:
            out = self._em.sendSnap(
                name=name,
                imgFrame=frame,
                data=data,
                tags=tags,
                extraData=extra_data,
                deviceSerialNo=device_serial_num,
            )
            if out:
                self._logger.info(f"Snap `{name}` sent")
                self.last_update = now
            return out
        return False


class SnapsProducer(SnapsProducerFrameOnly):
    """A host node that helps with creating and sending snaps. If you only have frame as
    input consider using `SnapsProducerFrameOnly` node instead.

    Attributes:
    ----------
    frame : dai.ImgFrame
        The frame input message for snap creation.
    msg : dai.Buffer
        The additional input message for snap creation.
    time_interval : float
        Time interval between snaps if no custom process function is provided. Defaults to 60s.
    process_fn : Callable[['SnapsProducer', dai.ImgFrame, dai.Buffer], None]
        Custom processing function for business logic regarding snaps. If None defaults to sending only frames on time intervals.
    token : str
        Hub API token of the team you want to save snaps under. Can be also set through `DEPTHAI_HUB_API_KEY` env variable.
    url : str
        Custom URL for events service. By default, the URL is set to https://events-ingest.cloud.luxonis.com
    """

    def setProcessFn(self, process_fn: ProcessFnType):
        """Sets custom processing function.

        @param process_fn: Custom snaps processing function.
        @type process_fn: Callable[['SnapsProducer', dai.ImgFrame, dai.Buffer], None]
        """
        if not isinstance(process_fn, Callable):
            raise ValueError("process_fn must be a function.")

        self._process_fn = process_fn
        self._logger.debug("Process function set")

    def build(
        self,
        frame: dai.Node.Output,
        msg: dai.Node.Output,
        time_interval: Union[int, float] = 60.0,
        process_fn: Optional[ProcessFnType] = None,
    ) -> "SnapsProducer":
        """Configures the node.

        @param frame: The frame input message for snap creation.
        @type frame: dai.Node.Output
        @param msg: The additonal input message for snap creation.
        @type msg: dai.Node.Output
        @param time_interval: Time interval between snaps for default sending in
            seconds. Defualts to 60.
        @type time_interval: Union[int, float]
        @param process_fn: Custom snaps processing function. Defaults to None.
        @type process_fn: Optional[Callable[['SnapsProducer', dai.ImgFrame, dai.Buffer],
            None]]
        @return: The node object which handles snap creation and sending.
        @rtype: SnapsProducer
        """
        self.link_args(frame, msg)
        self.setTimeInterval(time_interval)
        if process_fn is not None:
            self.setProcessFn(process_fn)
        self._logger.debug(f"SnapsProducer built with time_interval={time_interval}")
        return self

    def process(self, frame: dai.Buffer, msg: dai.Buffer):
        """Processes incoming frames and sends out snaps. If not custom process function
        is set then it sends only one frame every `time_interval` seconds. If using
        custom process function make sure to call `self.sendSnap()` at the end.

        @param frame: The frame input message for snap creation.
        @type frame: dai.ImgFrame
        @param frame: The additional input message for snap creation.
        @type frame: dai.Buffer
        """
        self._logger.debug("Processing new input")
        assert isinstance(frame, dai.ImgFrame)
        if self._process_fn is None:
            self.sendSnap("frame", frame)
        else:
            self._process_fn(self, frame, msg)
