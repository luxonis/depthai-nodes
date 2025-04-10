import depthai as dai
from abc import ABCMeta


HostNodeMeta = type(dai.node.HostNode)  # metaclass of dai.node.HostNode


class CombinedMeta(HostNodeMeta, ABCMeta):
    pass


class ImgFrameSender(dai.node.HostNode, metaclass=CombinedMeta):
    """An abstract base class for host nodes that send out dai.ImgFrame objects. Designed to encapsulate and abstract the configuration of platform-specific attributes,
    providing a clean and consistent interface for derived classes.

    """

    IMG_FRAME_TYPES = {
        "RVC2": dai.ImgFrame.Type.BGR888p,
        "RVC4": dai.ImgFrame.Type.BGR888i,
    }  # TODO: extend for other platforms?

    def __init__(self) -> None:
        super().__init__()

        self._platform = (
            self.getParentPipeline().getDefaultDevice().getPlatformAsString()
        )

        try:
            self._img_frame_type = self.IMG_FRAME_TYPES[self._platform]
        except KeyError:
            raise ValueError(
                f"No dai.ImgFrame.Type defined for platform {self._platform}."
            )

    def process(self) -> None:
        pass
