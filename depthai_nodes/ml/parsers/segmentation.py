from typing import Any, Dict

import depthai as dai
import numpy as np

from ..messages.creators import create_segmentation_message
from .base_parser import BaseParser


class SegmentationParser(BaseParser):
    """Parser class for parsing the output of the segmentation models.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    output_layer_name: str
        Name of the output layer from which the scores are extracted.
    background_class : bool
        Whether to add additional layer for background.

    Output Message/s
    ----------------
    **Type**: dai.ImgFrame

    **Description**: Segmentation message containing the segmentation mask. Every pixel belongs to exactly one class.

    Error Handling
    --------------
    **ValueError**: If the number of output layers is not E{1}.

    **ValueError**: If the number of dimensions of the output tensor is not E{3}.
    """

    def __init__(
        self, output_layer_name: str = "", background_class: bool = False
    ) -> None:
        """Initializes the SegmentationParser node.

        @param output_layer_name: Name of the output layer from which the scores are
            extracted.
        @type output_layer_name: str
        @param background_class: Whether to add additional layer for background.
        @type background_class: bool
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.background_class = background_class

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "SegmentationParser":
        """Sets the head configuration for the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.

        Returns
        -------
        SegmentationParser
            Returns the parser object with the head configuration set.
        """

        output_layers = head_config["outputs"]
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for Segmentation, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.background_class = head_config["background_class"]

        return self

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Sets the name of the output layer.

        @param output_layer_name: The name of the output layer.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name

    def setBackgroundClass(self, background_class: bool) -> None:
        """Sets the background class.

        @param background_class: Whether to add additional layer for background.
        @type background_class: bool
        """
        if not isinstance(background_class, bool):
            raise ValueError("Background class must be a boolean.")
        self.background_class = background_class

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            layers = output.getAllLayerNames()

            if len(layers) == 1 and self.output_layer_name == "":
                self.output_layer_name = layers[0]
            elif len(layers) != 1 and self.output_layer_name == "":
                raise ValueError(
                    f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
                )

            segmentation_mask = output.getTensor(
                self.output_layer_name, dequantize=True
            )
            if len(segmentation_mask.shape) == 4:
                segmentation_mask = segmentation_mask[0]

            if len(segmentation_mask.shape) != 3:
                raise ValueError(
                    f"Expected 3D output tensor, got {len(segmentation_mask.shape)}D."
                )

            mask_shape = segmentation_mask.shape
            min_dim = np.argmin(mask_shape)
            if min_dim == len(mask_shape) - 1:
                segmentation_mask = segmentation_mask.transpose(2, 0, 1)
            if self.background_class:
                segmentation_mask = np.vstack(
                    (
                        np.zeros(
                            (1, segmentation_mask.shape[1], segmentation_mask.shape[2]),
                            dtype=np.float32,
                        ),
                        segmentation_mask,
                    )
                )
            class_map = (
                np.argmax(segmentation_mask, axis=0)
                .reshape(segmentation_mask.shape[1], segmentation_mask.shape[2], 1)
                .astype(np.uint8)
            )

            imgFrame = create_segmentation_message(class_map)
            imgFrame.setTimestamp(output.getTimestamp())
            self.out.send(imgFrame)
