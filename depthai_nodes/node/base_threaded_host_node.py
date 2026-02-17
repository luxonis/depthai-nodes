from abc import ABCMeta, abstractmethod

import depthai as dai
from depthai_nodes.logging import get_logger

ThreadedHostNodeMeta = type(dai.node.ThreadedHostNode)  # metaclass of dai.node.HostNode


class CombinedMeta(ABCMeta, ThreadedHostNodeMeta):
    pass


class BaseThreadedHostNode(dai.node.ThreadedHostNode, metaclass=CombinedMeta):
    """An abstract base class for threaded host nodes.

    Designed to encapsulate and abstract the configuration of platform-specific
    attributes, providing a clean and consistent interface for derived classes.
    """

    IMG_FRAME_TYPES = {
        dai.Platform.RVC2: dai.ImgFrame.Type.BGR888p,
        dai.Platform.RVC4: dai.ImgFrame.Type.BGR888i,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = self.getParentPipeline()
        self._platform = self.getParentPipeline().getDefaultDevice().getPlatform()

        try:
            self._img_frame_type = self.IMG_FRAME_TYPES[self._platform]
        except KeyError as e:
            raise ValueError(
                f"No dai.ImgFrame.Type defined for platform {self._platform}."
            ) from e

        self._logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self) -> None:
        pass
