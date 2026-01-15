import depthai as dai

from depthai_nodes.logging import get_logger


class BaseThreadedHostNode(dai.node.ThreadedHostNode):
    """An abstract base class for host nodes.

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
            self._img_frame_type = self.IMG_FRAME_TYPES[self.platform]
        except KeyError as e:
            raise ValueError(
                f"No dai.ImgFrame.Type defined for platform {self.platform}."
            ) from e

        self._logger = get_logger(self.__class__.__name__)

    def run(self) -> None:
        pass
