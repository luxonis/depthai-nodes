from typing import Any, Dict

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_segmentation_message
from depthai_nodes.node.parsers.base_parser import BaseParser


class SegmentationParser(BaseParser):
    """Parser class for parsing the output of the segmentation models.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    classes_in_one_layer : bool
        Whether all classes are in one layer in the multi-class segmentation model. Default is False. If True, the parser will use np.max instead of np.argmax to get the class map.

    Output Message/s
    ----------------
    **Type**: SegmentationMask

    **Description**: Segmentation message containing the segmentation mask. Every pixel belongs to exactly one class. Unassigned pixels are represented with "-1" and class pixels with non-negative integers.

    Error Handling
    --------------
    **ValueError**: If the number of output layers is not E{1}.

    **ValueError**: If the number of dimensions of the output tensor is not E{3}.
    """

    def __init__(
        self, output_layer_name: str = "", classes_in_one_layer: bool = False
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param classes_in_one_layer: Whether all classes are in one layer in the multi-
            class segmentation model. Default is False. If True, the parser will use
            np.max instead of np.argmax to get the class map.
        @type classes_in_one_layer: bool
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.classes_in_one_layer = classes_in_one_layer
        self._logger.debug(
            f"SegmentationParser initialized with output_layer_name='{output_layer_name}', classes_in_one_layer={classes_in_one_layer}"
        )

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Sets the name of the output layer.

        @param output_layer_name: The name of the output layer.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to '{self.output_layer_name}'")

    def setClassesInOneLayer(self, classes_in_one_layer: bool) -> None:
        """Sets the flag indicating whether all classes are in one layer.

        @param classes_in_one_layer: Whether all classes are in one layer.
        @type classes_in_one_layer: bool
        """
        if not isinstance(classes_in_one_layer, bool):
            raise ValueError("classes_in_one_layer must be a boolean.")
        self.classes_in_one_layer = classes_in_one_layer
        self._logger.debug(f"Classes in one layer set to {self.classes_in_one_layer}")

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "SegmentationParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: SegmentationParser
        """

        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for Segmentation, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.classes_in_one_layer = head_config.get(
            "classes_in_one_layer", self.classes_in_one_layer
        )

        self._logger.debug(
            f"SegmentationParser built with output_layer_name='{self.output_layer_name}', classes_in_one_layer={self.classes_in_one_layer}"
        )

        return self

    def run(self):
        self._logger.debug("SegmentationParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            layers = output.getAllLayerNames()
            self._logger.debug(f"Processing input with layers: {layers}")
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

            np_function = np.argmax
            mask_shape = segmentation_mask.shape
            min_dim = np.argmin(mask_shape)
            if min_dim == len(mask_shape) - 1:
                segmentation_mask = segmentation_mask.transpose(2, 0, 1)
            adding_unassigned_class = False
            if segmentation_mask.shape[0] == 1:  # shape is (1, H, W)
                if self.classes_in_one_layer:
                    np_function = np.max
                else:
                    # If there is only one class, add an unassigned class
                    adding_unassigned_class = True
                    segmentation_mask = np.vstack(
                        (
                            np.zeros(
                                (
                                    1,
                                    segmentation_mask.shape[1],
                                    segmentation_mask.shape[2],
                                ),
                                dtype=np.float32,
                            ),
                            segmentation_mask,
                        )
                    )

            class_map = (
                np_function(segmentation_mask, axis=0)
                .reshape(segmentation_mask.shape[1], segmentation_mask.shape[2])
                .astype(np.int16)
            )

            if adding_unassigned_class:
                class_map = class_map - 1

            mask_message = create_segmentation_message(class_map)
            mask_message.setTimestamp(output.getTimestamp())
            mask_message.setSequenceNum(output.getSequenceNum())
            mask_message.setTimestampDevice(output.getTimestampDevice())
            transformation = output.getTransformation()
            if transformation is not None:
                mask_message.setTransformation(transformation)

            self._logger.debug(
                f"Created segmentation message with {class_map.shape[0]} classes"
            )

            self.out.send(mask_message)

            self._logger.debug("Segmentation message sent successfully")
