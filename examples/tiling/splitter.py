import depthai as dai

from depthai_nodes.message import GatheredData
from depthai_nodes.node.base_host_node import BaseHostNode


class SplitterNode(BaseHostNode):
    """Merge all ImgDetections stored in a GatheredData message into one message."""
    def __init__(self):
        super().__init__()
        self.seq_num = -1
        self.outputs = []
        self.counter = 0

    def build(self, input: dai.Node.Output, nr_outs: int) -> "MergeImgDetections":
        self.link_args(input)
        for i in range(nr_outs):
            self.outputs.append(self.createOutput(name=f"out_{i}"))
        return self

    def process(self, msg: dai.Buffer) -> None:
        new_seq_num = msg.getSequenceNum()
        if new_seq_num != self.seq_num:
            self.seq_num = new_seq_num
            self.counter = 0
        self.outputs[self.counter].send(msg)
        self.counter += 1
