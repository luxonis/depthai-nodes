from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_segmentation_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.segmentation import (
    compute_segmentation_class_map,
)


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
    **Type**: dai.SegmentationMask

    **Description**: dai.SegmentationMask containing the segmentation mask. Every pixel belongs to exactly one class. Unassigned pixels are represented with "255" and class pixels with non-negative integers.

    Error Handling
    --------------
    **ValueError**: If the number of output layers is not E{1}.

    **ValueError**: If the number of dimensions of the output tensor is not E{3}.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        classes_in_one_layer: bool = False,
        background_class: bool = False,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param classes_in_one_layer: Whether all classes are in one layer in the multi-
            class segmentation model. Default is False. If True, the parser will use
            np.max instead of np.argmax to get the class map.
        @type classes_in_one_layer: bool
        @param background_class: Whether class index 0 should be treated as background.
        @type background_class: bool
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.classes_in_one_layer = classes_in_one_layer
        self.background_class = background_class
        self.class_names = None
        self.n_classes = 0
        self._background_class_ignored_warning_sent = False
        self._logger.debug(
            "SegmentationParser initialized with "
            f"output_layer_name='{output_layer_name}', "
            f"classes_in_one_layer={classes_in_one_layer}, "
            f"background_class={background_class}"
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

    def setBackgroundClass(self, background_class: bool) -> None:
        """Sets whether class index 0 should be treated as background.

        @param background_class: Whether class index 0 is background.
        @type background_class: bool
        """
        if not isinstance(background_class, bool):
            raise ValueError("background_class must be a boolean.")
        self.background_class = background_class
        self._logger.debug(f"Background class set to {self.background_class}")

    def _warn_if_background_class_ignored(self) -> None:
        if (
            self.background_class
            and self.classes_in_one_layer
            and not self._background_class_ignored_warning_sent
        ):
            self._logger.warning(
                "background_class=True is ignored when classes_in_one_layer=True."
            )
            self._background_class_ignored_warning_sent = True

    def _get_logged_class_count(self, class_map: np.ndarray) -> int:
        if self.class_names is not None:
            return len(self.class_names)
        if self.n_classes > 0:
            return self.n_classes
        return int(np.unique(class_map[class_map != 255]).size)

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "SegmentationParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: dict[str, Any]
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
        self.background_class = head_config.get(
            "background_class", self.background_class
        )
        self.class_names = head_config.get("classes", self.class_names)
        self.n_classes = head_config.get("n_classes", self.n_classes)
        if self.n_classes == 0 and self.class_names is not None:
            self.n_classes = len(self.class_names)

        self._logger.debug(
            "SegmentationParser built with "
            f"output_layer_name='{self.output_layer_name}', "
            f"classes_in_one_layer={self.classes_in_one_layer}, "
            f"background_class={self.background_class}"
        )

        self._warn_if_background_class_ignored()

        return self

    def run(self):
        self._logger.debug("SegmentationParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            segmentation_mask = self.extract(output)
            class_map = self.compute(
                segmentation_mask,
                classes_in_one_layer=self.classes_in_one_layer,
                background_class=self.background_class,
            )
            self.emit(output, class_map)

    def extract(self, output: dai.NNData):
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
    def compute(
        segmentation_mask: np.ndarray,
        *,
        classes_in_one_layer: bool = False,
        background_class: bool = False,
    ) -> np.ndarray:
        return compute_segmentation_class_map(
            segmentation_mask,
            classes_in_one_layer=classes_in_one_layer,
            background_class=background_class,
        )

    def emit(self, output: dai.NNData, class_map: np.ndarray) -> None:
        mask_message = create_segmentation_message(class_map)
        mask_message.setTimestamp(output.getTimestamp())
        mask_message.setSequenceNum(output.getSequenceNum())
        mask_message.setTimestampDevice(output.getTimestampDevice())
        transformation = output.getTransformation()
        if transformation is not None:
            mask_message.setTransformation(transformation)

        self._logger.debug(
            f"Created segmentation message with {self._get_logged_class_count(class_map)} classes"
        )
        self.out.send(mask_message)
        self._logger.debug("Segmentation message sent successfully")
