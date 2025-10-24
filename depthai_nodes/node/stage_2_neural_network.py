from typing import Union

import depthai as dai

from depthai_nodes.message.gathered_data import GatheredData
from depthai_nodes.node.base_host_node import BaseHostNode
from depthai_nodes.node.detection_cropper import DetectionCropper
from depthai_nodes.node.gather_data import GatherData
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node.utils.detection_remapping import remap_message


class Stage2NeuralNetwork(BaseHostNode):
    """This node handles cropping detections from a stage 1 neural network, resizes them
    and inputs them to another neural network. It can also remap detection coordinates
    to match input frame transformations. Supports only ImgDetections and
    ImgDetectionsExtended messages for stage 1 neural network.

    Attributes
    ----------
    out : Node.Output
        Neural network output. Detections are remapped to the input frame coordinates if remap_detections is True.
    nn_passthrough : Node.Output
        Neural network frame passthrough.
    """

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

    def __init__(self):
        super().__init__()

        self._padding = 0.0

        self._pipeline = self.getParentPipeline()

        if self._platform == dai.Platform.RVC2:
            raise RuntimeError(
                "Stage2NeuralNetwork node is currently not supported on RVC2."
            )

        self._remap_detections = False
        self.nn = self._pipeline.create(ParsingNeuralNetwork)
        self.detection_cropper = self._pipeline.create(DetectionCropper)
        self.gather_data = self._pipeline.create(GatherData)
        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(self.SCRIPT_CONTENT)
        self._logger.debug("Stage2NeuralNetwork initialized")

    def build(
        self,
        img_frame: dai.Node.Output,
        stage_1_nn: dai.Node.Output,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
        fps: int,
        remap_detections: bool = False,
    ) -> "Stage2NeuralNetwork":
        """Configures the Stage2NeuralNetwork node and links all outputs.

        @param img_frame: The input frame. Crops will be performed on this frame.
        @type img_frame: dai.Node.Output
        @param stage_1_nn: The output of the stage 1 neural network node. Accepts ImgDetections and
        ImgDetectionsExtended messages.
        @type stage_1_nn: dai.Node.Output
        @param nn_source: NNModelDescription object containing the HubAI model descriptors, NNArchive object of the model, or HubAI model slug in form of <model_slug>:<model_version_slug> or <model_slug>:<model_version_slug>:<model_instance_hash>.
        @type nn_source: Union[dai.NNModelDescription, dai.NNArchive, str]
        @param input_resize_mode: How to resize the input crops.
        @type input_resize_mode: dai.ImageManipConfig.ResizeMode
        @param fps: Camera FPS.
        @type fps: int
        @param remap_detections: If True, remaps detection coordinates from neural network output to input frame coordinates.
        @type remap_detections: bool
        @return: Returns self for method chaining.
        @rtype: TilesPatcher
        """
        self._remap_detections = remap_detections

        self.nn.build(self.detection_cropper.out, nn_source)
        nn_size = self.nn._nn_archive.getInputSize()

        if nn_size is None:
            raise ValueError("NNArchive does not contain input size")

        self.detection_cropper.build(
            stage_1_nn,
            img_frame,
            nn_size,
            input_resize_mode,
            padding=self._padding,
        )
        self.gather_data.build(fps)

        try:
            self.nn.out.link(self.gather_data.input_data)
        except RuntimeError:
            # Model has multiple outputs
            self.nn.outputs.link(self.gather_data.input_data)
        stage_1_nn.link(self.gather_data.input_reference)
        img_frame.link(self._script.inputs["preview"])
        self.link_args(self.gather_data.out, self._script.outputs["transformation"])
        return self

    def _remapDetections(
        self, gathered_data: GatheredData, dst_transformation: dai.ImgTransformation
    ) -> GatheredData:
        remapped_msgs = [
            self._remapMessage(msg, gathered_data.reference_data.getTransformation())
            for msg in gathered_data.gathered
        ]
        new_gathered_data = GatheredData(gathered_data.reference_data, remapped_msgs)
        new_gathered_data.setTimestamp(gathered_data.getTimestamp())
        new_gathered_data.setSequenceNum(gathered_data.getSequenceNum())
        new_gathered_data.setTimestampDevice(gathered_data.getTimestampDevice())
        return new_gathered_data

    def _remapMessage(self, msg: dai.Buffer, dst_transformation: dai.ImgTransformation):
        if isinstance(msg, dai.MessageGroup):
            new_msg_group = dai.MessageGroup()
            for name in msg.getMessageNames():
                new_msg = self._remapMessage(msg[name], dst_transformation)
                new_msg_group[name] = new_msg
            new_msg_group.setTimestamp(msg.getTimestamp())
            new_msg_group.setSequenceNum(msg.getSequenceNum())
            new_msg_group.setTimestampDevice(msg.getTimestampDevice())
            return new_msg_group
        try:
            remapped_msg = remap_message(
                msg.getTransformation(), dst_transformation, msg
            )
        except Exception:
            remapped_msg = msg
        remapped_msg.setTransformation(dst_transformation)
        remapped_msg.setTimestamp(msg.getTimestamp())
        remapped_msg.setSequenceNum(msg.getSequenceNum())
        remapped_msg.setTimestampDevice(msg.getTimestampDevice())
        return remapped_msg

    def process(self, gathered_data: dai.Buffer, img_frame: dai.Buffer) -> None:
        assert isinstance(img_frame, dai.ImgFrame)
        assert isinstance(gathered_data, GatheredData)
        msg = gathered_data
        if self._remap_detections:
            msg = self._remapDetections(gathered_data, img_frame.getTransformation())
        self.out.send(msg)

    def setPadding(self, padding: float) -> None:
        self.detection_cropper.setPadding(padding)

    @property
    def nn_passthrough(self):
        return self.nn.passthrough
