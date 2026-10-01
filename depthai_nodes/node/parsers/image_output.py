from typing import Any

import depthai as dai

from depthai_nodes.message.creators import create_image_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.image_output import compute_image_output


class ImageOutputParser(BaseParser):
    """Parser class for image-to-image models (e.g. DnCNN3, zero-dce etc.) where the
    output is a modified image (denoised, enhanced etc.).

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.
        output_is_bgr (``bool``): Flag indicating if the output image is in BGR
            (Blue-Green-Red) format.

    Note:
        Emits ``dai.ImgFrame`` messages. Image message containing the output image e.g.
        denoised or enhanced images.

    Raises:
        ValueError: If the output is not 3- or 4-dimensional.

        ValueError: If the number of output layers is not 1.
    """

    def __init__(
        self, output_layer_name: str = "", output_is_bgr: bool = False
    ) -> None:
        """Initialize the parser node.

        Args:
            output_layer_name: Output tensor name. An empty name selects the only
                available output layer during extraction.
            output_is_bgr: Whether the output image uses BGR channel order.
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.output_is_bgr = output_is_bgr

        self._platform = (
            self.getParentPipeline().getDefaultDevice().getPlatformAsString()
        )
        self._logger.debug(
            f"ImageOutputParser initialized with output_layer_name='{output_layer_name}', output_is_bgr={output_is_bgr}"
        )

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Sets the name of the output layer.

        Args:
            output_layer_name: The name of the output layer.
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to '{self.output_layer_name}'")

    def setBGROutput(self) -> None:
        """Sets the flag indicating that output image is in BGR."""
        self.output_is_bgr = True
        self._logger.debug(f"Output is BGR set to {self.output_is_bgr}")

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "ImageOutputParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 1:
            raise ValueError(
                f"MapOutputParser expects exactly 1 output layers, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.output_is_bgr = head_config.get("output_is_bgr", self.output_is_bgr)

        self._logger.debug(
            f"ImageOutputParser built with output_layer_name='{self.output_layer_name}', output_is_bgr={self.output_is_bgr}"
        )

        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("ImageOutputParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            output_image = self.extract(output)
            image = self.compute(output_image)
            self.emit(output, image)

    def extract(self, output: dai.NNData):
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Dequantized image tensor retaining the model tensor layout.

        Raises:
            ValueError: If no output name is configured and the message does not contain
                exactly one layer, or configured class requirements are not met.
        """
        layers = output.getAllLayerNames()
        self._logger.debug(f"Processing input with layers: {layers}")
        if len(layers) == 1 and self.output_layer_name == "":
            self.output_layer_name = layers[0]
        elif len(layers) != 1 and self.output_layer_name == "":
            raise ValueError(
                f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
            )

        return output.getTensor(self.output_layer_name, dequantize=True)

    @staticmethod
    def compute(output_image):
        """Compute parser results from extracted tensors without sending messages.

        Args:
            output_image: CHW or HWC tensor, optionally preceded by a singleton batch
                dimension.

        Returns:
            A uint8 array with the same channel layout as the unbatched input. Values
            are min-max scaled to [0, 255]; constant tensors become zero.

        Note:
            Uses `depthai_nodes.node.parsers.utils.image_output.compute_image_output`;
            see that helper for tensor layout and validation details.
        """
        return compute_image_output(output_image)

    def emit(self, output: dai.NNData, image) -> None:
        """Create a ``dai.ImgFrame`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            image: Image array returned by ``compute()``.
        """
        image_message = create_image_message(
            image=image,
            is_bgr=self.output_is_bgr,
            img_frame_type=(
                dai.ImgFrame.Type.BGR888p
                if self._platform == "RVC2"
                else dai.ImgFrame.Type.BGR888i
            ),
        )
        image_message.setTimestamp(output.getTimestamp())
        image_message.setSequenceNum(output.getSequenceNum())
        image_message.setTimestampDevice(output.getTimestampDevice())
        transformation = output.getTransformation()
        if transformation is not None:
            image_message.setTransformation(transformation)

        self._logger.debug(f"Created image message with shape {image.shape}")
        self.out.send(image_message)
        self._logger.debug("Image message sent successfully")
