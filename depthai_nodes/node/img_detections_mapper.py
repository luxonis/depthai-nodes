import depthai as dai

from depthai_nodes.message.img_detections import (
    ImgDetectionsExtended,
)
from depthai_nodes.node.base_host_node import BaseHostNode
from depthai_nodes.node.utils.detection_remapping import remap_message

UNASSIGNED_MASK_LABEL = -1


class ImgDetectionsMapper(BaseHostNode):
    """Remap ImgDetections to ImgFrame coordinates."""

    SCRIPT_CONTENT = """
# Strip ImgFrame image data and send only ImgTransformation
# Reduces the amount of date being sent between host and device

try:
    while True:
        frame = node.inputs['preview'].get()
        transformation = frame.getTransformation()
        empty_frame = ImgFrame()
        empty_frame.setTransformation(transformation)
        empty_frame.setTimestamp(frame.getTimestamp())
        empty_frame.setTimestampDevice(frame.getTimestampDevice())
        node.outputs['transformation'].send(empty_frame)
except Exception as e:
    node.warn(str(e))
"""

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = self.getParentPipeline()
        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(self.SCRIPT_CONTENT)
        self._logger.debug("ImgDetectionsMapper initialized")

    def build(
        self, img_input: dai.Node.Output, nn_input: dai.Node.Output
    ) -> "ImgDetectionsMapper":
        img_input.link(self._script.inputs["preview"])
        self._script.outputs["transformation"].setPossibleDatatypes(
            [(dai.DatatypeEnum.ImgFrame, True)]
        )
        self.link_args(self._script.outputs["transformation"], nn_input)
        return self

    def process(self, img: dai.ImgFrame, nn: dai.Buffer) -> None:
        assert isinstance(
            nn, (dai.ImgDetections, ImgDetectionsExtended)
        ), "Expected ImgDetections or ImgDetectionsExtended"

        nn_trans = nn.getTransformation()
        if nn_trans is None:
            raise RuntimeError("Received detection message without transformation")
        message = remap_message(nn_trans, img.getTransformation(), nn)
        message.setTimestamp(nn.getTimestamp())
        message.setTimestampDevice(nn.getTimestampDevice())
        message.setSequenceNum(nn.getSequenceNum())
        message.setTransformation(img.getTransformation())
        self.out.send(message)
