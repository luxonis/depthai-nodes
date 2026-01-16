import depthai as dai

from depthai_nodes.node.base_host_node import BaseHostNode
from depthai_nodes.node.utils.message_remapping import remap_message


class CoordinatesMapper(BaseHostNode):
    """Remap coordinates to different reference frame.

    Can handle any depthai message that returns dai.ImgTransformation from `getTransformation()` method.
    """

    SCRIPT_CONTENT = """
# Strip ImgFrame image data and send only ImgTransformation
# Reduces the amount of date being sent between host and device

try:
    while True:
        message = node.inputs['message'].get()
        transformation = message.getTransformation()
        empty_frame = ImgFrame()
        empty_frame.setTransformation(transformation)
        empty_frame.setTimestamp(message.getTimestamp())
        empty_frame.setTimestampDevice(message.getTimestampDevice())
        node.outputs['transformation'].send(empty_frame)
except Exception as e:
    node.warn(str(e))
"""

    def __init__(self) -> None:
        super().__init__()
        if self._platform == dai.Platform.RVC2:
            raise RuntimeError(
                "CoordinatesMapper node is currently not supported on RVC2."
            )

    def build(
        self, to_transformation_input: dai.Node.Output, from_transformation_input: dai.Node.Output
    ) -> "CoordinatesMapper":
        script = self._pipeline.create(dai.node.Script)
        script.setScript(self.SCRIPT_CONTENT)
        to_transformation_input.link(script.inputs["message"])
        script.outputs["transformation"].setPossibleDatatypes(
            [(dai.DatatypeEnum.ImgFrame, True)]
        )
        self.link_args(script.outputs["transformation"], from_transformation_input)
        self._logger.debug("CoordinatesMapper built")
        return self

    def process(self, to_transformation_msg: dai.ImgFrame, from_transformation_msg: dai.Buffer) -> None:
        try:
            to_transformation: dai.ImgTransformation = to_transformation_msg.getTransformation()
        except Exception as e:
            raise RuntimeError(
                "Could not get transformation from `to_transformation_msg` message. Message doesn't have the `getTransformation()` method."
            ) from e
        if to_transformation is None:
            raise RuntimeError("Received `to_transformation_msg` message without transformation. The `getTransformation()` method returns None.")
        remapped_message = self._remap_message(msg=from_transformation_msg, to_transformation=to_transformation)
        self.out.send(remapped_message)

    def _remap_message(self, msg: dai.Buffer, to_transformation: dai.ImgTransformation):
        if isinstance(msg, dai.MessageGroup):
            new_msg_group = dai.MessageGroup()
            for name in msg.getMessageNames():
                new_msg = self._remap_message(msg[name], to_transformation)
                new_msg_group[name] = new_msg
            new_msg_group.setTimestamp(msg.getTimestamp())
            new_msg_group.setSequenceNum(msg.getSequenceNum())
            new_msg_group.setTimestampDevice(msg.getTimestampDevice())
            return new_msg_group
        try:
            remapped_msg = remap_message(
                message=msg,
                from_transformation=msg.getTransformation(),
                to_transformation=to_transformation,
            )
        except Exception:
            remapped_msg = msg
        remapped_msg.setTransformation(to_transformation)
        remapped_msg.setTimestamp(msg.getTimestamp())
        remapped_msg.setSequenceNum(msg.getSequenceNum())
        remapped_msg.setTimestampDevice(msg.getTimestampDevice())
        return remapped_msg
