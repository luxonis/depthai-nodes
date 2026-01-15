from typing import Union

import depthai as dai

from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.coordinates_mapper import CoordinatesMapper
from depthai_nodes.node.detection_cropper import DetectionCropper
from depthai_nodes.node.gather_data import GatherData
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node.utils.message_remapping import remap_message


class Stage2NeuralNetwork(BaseThreadedHostNode):
    """This node handles cropping detections from a stage 1 neural network, resizes them
    and inputs them to another neural network. It can also remap detection coordinates
    to match input frame transformations. Supports only ImgDetections and
    ImgDetectionsExtended messages for stage 1 neural network.

    Properties
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

        self.nn: ParsingNeuralNetwork = self._pipeline.create(ParsingNeuralNetwork)
        self.detection_cropper: DetectionCropper = self._pipeline.create(DetectionCropper)
        self.gather_data: GatherData = self._pipeline.create(GatherData)
        self._logger.debug("Stage2NeuralNetwork initialized")

    def build(
        self,
        img_frame: dai.Node.Output,
        stage_1_nn: dai.Node.Output,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
        fps: int,
        remap_detections: bool = True,
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
        self.nn.build(self.detection_cropper.out, nn_source)
        nn_size = self.nn._nn_archive.getInputSize()
        if nn_size is None:
            raise ValueError("NNArchive does not contain input size")
        try:
            nn_output = self.nn.out
        except RuntimeError:
            # Model has multiple outputs
            nn_output = self.nn.outputs
        self.detection_cropper.build(
            stage_1_nn,
            img_frame,
            nn_size,
            input_resize_mode,
            padding=0.0,
        )
        gather_node_data_input = nn_output

        if remap_detections:
            script = self._pipeline.create(dai.node.Script)
            script.setScript(self.SCRIPT_CONTENT)
            img_frame.link(script.inputs["preview"])
            coordinates_mapper = self._pipeline.create(CoordinatesMapper).build(
                to_transformation_input=script.outputs["transformation"],
                from_transformation_input=nn_output,
            )
            gather_node_data_input = coordinates_mapper.out

        self.gather_data.build(
            camera_fps=fps,
            input_data=gather_node_data_input,
            input_reference=stage_1_nn,
        )
        return self

    def _remapMessage(self, msg: dai.Buffer, to_transformation: dai.ImgTransformation):
        if isinstance(msg, dai.MessageGroup):
            new_msg_group = dai.MessageGroup()
            for name in msg.getMessageNames():
                new_msg = self._remapMessage(msg[name], to_transformation)
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

    def run(self) -> None:
        pass

    def setPadding(self, padding: float) -> None:
        self.detection_cropper.setPadding(padding)

    @property
    def out(self):
        return self.gather_data.out

    @property
    def nn_passthrough(self):
        return self.nn.passthrough
