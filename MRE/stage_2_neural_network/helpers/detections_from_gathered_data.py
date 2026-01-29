import depthai as dai

from depthai_nodes.message import GatheredData, ImgDetectionsExtended
from depthai_nodes.node.utils.message_merging import merge_messages


class DetectionsFromGatheredData(dai.node.HostNode):
    """"""

    def build(self, input: dai.Node.Output) -> "DetectionsFromGatheredData":
        self.link_args(input)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, gathered_data: dai.Buffer) -> None:
        assert isinstance(gathered_data, GatheredData)
        gathered_data: GatheredData
        gathered: list[ImgDetectionsExtended] = gathered_data.gathered
        if gathered:
            detections = merge_messages(messages=gathered)
            reference: ImgDetectionsExtended = gathered_data.reference_data
            detections.setTimestampDevice(reference.getTimestampDevice())
            detections.setTimestamp(reference.getTimestamp())
            detections.setTransformation(reference.getTransformation())
            self.out.send(detections)
