from typing import Any, Dict

import depthai as dai

from ..messages.creators import create_image_message
from .base_parser import BaseParser
from .utils import unnormalize_image


class ImageOutputParser(BaseParser):
    """Parser class for image-to-image models (e.g. DnCNN3, zero-dce etc.) where the
    output is a modifed image (denoised, enhanced etc.).

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    output_layer_name: str
        Name of the output layer from which the image output is extracted.
    output_is_bgr : bool
        Flag indicating if the output image is in BGR (Blue-Green-Red) format.

    Output Message/s
    -------
    **Type**: dai.ImgFrame

    **Description**: Image message containing the output image e.g. denoised or enhanced images.

    Error Handling
    --------------
    **ValueError**: If the output is not 3- or 4-dimensional.

    **ValueError**: If the number of output layers is not 1.
    """

    def __init__(
        self, output_layer_name: str = "", output_is_bgr: bool = False
    ) -> None:
        """Initializes ImageOutputParser node.

        @param output_is_bgr: Flag indicating if the output image is in BGR.
        @type output_is_bgr: bool
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.output_is_bgr = output_is_bgr

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "ImageOutputParser":
        """Sets the head configuration for the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.

        Returns
        -------
        ImageOutputParser
            Returns the parser object with the head configuration set.
        """

        output_layers = head_config["outputs"]
        if len(output_layers) != 1:
            raise ValueError(
                f"MapOutputParser expects exactly 1 output layers, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.output_is_bgr = head_config.get("output_is_bgr", self.output_is_bgr)

        return self

    def setBGROutput(self) -> None:
        """Sets the flag indicating that output image is in BGR."""
        self.output_is_bgr = True

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            output_image = output.getTensor(self.output_layer_name, dequantize=True)

            if output_image.shape[0] == 1:
                output_image = output_image[0]  # remove batch dimension

            if len(output_image.shape) != 3:
                raise ValueError(
                    f"Expected 3D output tensor, got {len(output_image.shape)}D."
                )

            image = unnormalize_image(output_image)

            image_message = create_image_message(
                image=image,
                is_bgr=self.output_is_bgr,
            )
            image_message.setTimestamp(output.getTimestamp())

            self.out.send(image_message)
