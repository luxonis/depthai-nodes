from typing import Optional, Union

import depthai as dai

from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.coordinates_mapper import CoordinatesMapper
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork


class ExtendedNeuralNetwork(BaseThreadedHostNode):
    """A high-level host node that performs neural network inference with
    automatic input resizing and optional coordinate remapping.

    `ExtendedNeuralNetwork` is a convenience wrapper around an internal
    :class:`ParsingNeuralNetwork` node. It handles:

    - Model loading from HubAI slug, :class:`dai.NNModelDescription`,
      or :class:`dai.NNArchive`.
    - Automatic input resizing to match the neural network input resolution.
    - Optional coordinate remapping when the input is not a camera node.

    Two input modes are supported:

    - **Camera input**: When `inputImage` is a :class:`dai.node.Camera`,
      the node requests a resized output directly from the camera using
      the appropriate hardware resize mode. In this case, the neural
      network outputs are already aligned with the original image
      coordinates and no additional mapping is required.

    - **Generic stream input**: When `inputImage` is a
      :class:`dai.Node.Output`, an internal :class:`dai.node.ImageManip`
      node resizes frames to the network's expected input size. A
      :class:`CoordinatesMapper` node is then inserted to map neural
      network outputs back to the original image coordinate space.

    The node exposes neural network outputs via :attr:`out`, and
    passthrough frames via :attr:`passthrough`.

    Notes
    -----
    - This node is currently not supported on the RVC2 platform.
    - When a non-camera input is used, an additional ImageManip node
      is inserted into the pipeline.
    - Coordinate remapping is performed automatically when resizing
      occurs outside of a camera node.

    Outputs
    -------
    out : dai.Node.Output
        Parsed neural network output stream. If coordinate remapping
        is required, this stream contains remapped results.
    outputs : dai.Node.Output
        Alias for :attr:`out` or the raw neural network outputs,
        depending on input mode.
    passthrough : dai.Node.Output
        Passthrough stream from the underlying neural network node.

    See Also
    --------
    ParsingNeuralNetwork
        Node responsible for running inference and parsing results.
    CoordinatesMapper
        Node used to remap output coordinates when resizing is applied.
    dai.node.ImageManip
        Node used for resizing when input is not a camera node.
    """

    _RESIZE_MODE_MAP = {
        dai.ImageManipConfig.ResizeMode.CENTER_CROP: dai.ImgResizeMode.CROP,
        dai.ImageManipConfig.ResizeMode.LETTERBOX: dai.ImgResizeMode.LETTERBOX,
        dai.ImageManipConfig.ResizeMode.STRETCH: dai.ImgResizeMode.STRETCH,
        dai.ImageManipConfig.ResizeMode.NONE: dai.ImgResizeMode.STRETCH,
    }

    def __init__(self) -> None:
        super().__init__()
        if self._platform == dai.Platform.RVC2:
            raise RuntimeError(
                "ExtendedNeuralNetwork node is currently not supported on RVC2."
            )
        self._has_camera_node_as_input = False
        self._nn: Optional[ParsingNeuralNetwork] = None
        self._coordinates_mapper: Optional[CoordinatesMapper] = None

    @property
    def out(self) -> Optional[dai.Node.Output]:
        if not self._nn:
            return None
        if self._has_camera_node_as_input:
            return self._nn.out
        else:
            return self._coordinates_mapper.out

    @property
    def outputs(self) -> Optional[dai.Node.Output]:
        if not self._nn:
            return None
        if self._has_camera_node_as_input:
            return self._nn.outputs
        else:
            return self._coordinates_mapper.out

    @property
    def passthrough(self) -> Optional[dai.Node.Output]:
        if not self._nn:
            return None
        return self._nn.passthrough

    def build(
            self,
            inputImage: dai.node.Camera | dai.Node.Output,
            nnSource: Union[dai.NNModelDescription, dai.NNArchive, str],
            resizeMode: dai.ImageManipConfig.ResizeMode = dai.ImageManipConfig.ResizeMode.CENTER_CROP,
    ) -> "ExtendedNeuralNetwork":
        """Build the internal neural network pipeline.

                Configures model loading, input resizing, and optional coordinate
                remapping. Returns `self` for fluent chaining.

                Parameters
                ----------
                inputImage : dai.node.Camera or dai.Node.Output
                    Source of input frames. If a Camera node is provided,
                    resizing is requested directly from the camera. Otherwise,
                    an internal ImageManip node performs resizing.
                nnSource : Union[dai.NNModelDescription, dai.NNArchive, str]
                    Neural network source specification. Can be:

                    - A HubAI model slug (``str``),
                    - A :class:`dai.NNModelDescription`,
                    - A preloaded :class:`dai.NNArchive`.

                resizeMode : dai.ImageManipConfig.ResizeMode, optional
                    Resize strategy used when adapting input frames to the neural
                    network input size. Default is ``CENTER_CROP``.

                Returns
                -------
                ExtendedNeuralNetwork
                    The configured node instance.

                Raises
                ------
                RuntimeError
                    If the node is used on the RVC2 platform.
                ValueError
                    If `nnSource` is not a supported type.

                Notes
                -----
                - The neural network input resolution is inferred from the
                  provided model archive.
                - When `inputImage` is not a Camera node, an internal
                  ImageManip node resizes frames to match the model's
                  expected input size.
                - In non-camera mode, a CoordinatesMapper node remaps
                  inference results back to the original image space.
                """

        if isinstance(nnSource, str):
            nnSource = dai.NNModelDescription(nnSource)
        if isinstance(nnSource, dai.NNModelDescription):
            if not nnSource.platform:
                nnSource.platform = self._platform.name
            nn_archive = dai.NNArchive(dai.getModelFromZoo(nnSource))
        elif isinstance(nnSource, dai.NNArchive):
            nn_archive = nnSource
        else:
            raise ValueError(
                "nn_source must be either a NNModelDescription, NNArchive, or a string representing HubAI model slug."
            )
        nn_w = nn_archive.getInputWidth()
        nn_h = nn_archive.getInputHeight()
        if isinstance(inputImage, dai.node.Camera):
            image_out = inputImage.requestOutput(
                size=(nn_w, nn_h),
                type=self.IMG_FRAME_TYPES[self._platform],
                resizeMode=self._RESIZE_MODE_MAP[resizeMode],
            )
            self._has_camera_node_as_input = True
        else:
            manip = self._pipeline.create(dai.node.ImageManip)
            manip.setMaxOutputFrameSize(nn_w * nn_h * 3)
            manip.initialConfig.setFrameType(self.IMG_FRAME_TYPES[self._platform])
            manip.initialConfig.setOutputSize(w=nn_w, h=nn_h, mode=resizeMode)
            inputImage.link(manip.inputImage)
            image_out = manip.out

        self._nn = self._pipeline.create(ParsingNeuralNetwork).build(input=image_out, nn_source=nn_archive)

        try:
            nn_output = self._nn.out
        except RuntimeError:
            nn_output = self._nn.outputs

        if not self._has_camera_node_as_input:
            self._coordinates_mapper = self._pipeline.create(CoordinatesMapper).build(
                toTransformationInput=inputImage,
                fromTransformationInput=nn_output,
            )

        self._logger.debug("ExtendedNeuralNetwork built")
        return self

    def run(self):
        pass
