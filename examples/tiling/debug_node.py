import depthai as dai

from depthai_nodes.message import GatheredData
from depthai_nodes.node.base_host_node import BaseHostNode


class DebugNode(BaseHostNode):
    """Merge all ImgDetections stored in a GatheredData message into one message."""

    def build(self, input: dai.Node.Output) -> "MergeImgDetections":
        self.link_args(input)
        return self

    def process(self, msg: dai.Buffer) -> None:
        # print(f"DEBUG: {msg.getSequenceNum()=}")
        pass
