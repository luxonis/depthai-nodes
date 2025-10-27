from typing import List, Dict, Any

import depthai as dai

from depthai_nodes.message.creators import create_classification_message
from depthai_nodes.node.parsers.base_parser import BaseParser
import numpy as np
from depthai_nodes.node.parsers.utils import softmax


class TransformerClassificationParser(BaseParser):

    def __init__(self,
                 output_layer_name: str = "",
                 classes: List[str] = None,
                 ) -> None:
        super().__init__()
        self.output_layer_name = output_layer_name
        self.classes = classes or []
        self.n_classes = len(self.classes)
        self._logger.debug(
            f"TransformerClassificationParser initialized"
        )

    def setClasses(self, classes: List[str]) -> None:
        """Sets the class names for the classification model.

        @param classes: List of class names to be used for linking with their respective
            scores.
        @type classes: List[str]
        """
        if not isinstance(classes, list):
            raise ValueError("classes must be a list.")
        for class_name in classes:
            if not isinstance(class_name, str):
                raise ValueError("Each class name must be a string.")
        self.classes = classes if classes is not None else []
        self.n_classes = len(self.classes)
        self._logger.debug(f"Classes set to {self.classes}")

    def build(self, head_config: Dict[str, Any]) -> "TransformerClassificationParser":
        self._logger.debug(head_config)
        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for Classification, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.classes = head_config.get("classes", self.classes)
        self.n_classes = head_config.get("n_classes", self.n_classes)

        self._logger.debug(
            f"TransformerClassificationParser built with output_layer_name='{self.output_layer_name}', classes={self.classes}, n_classes={self.n_classes}"
        )
        return self

    def run(self):
        self._logger.debug("TransformerClassificationParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break

            layers = output.getAllLayerNames()
            self._logger.debug(f"Processing input with layers: {layers}")
            if len(layers) == 1 and self.output_layer_name == "":
                self.output_layer_name = layers[0]
            elif len(layers) != 1 and self.output_layer_name == "":
                raise ValueError(
                    f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
                )

            scores = output.getTensor(self.output_layer_name, dequantize=True)
            scores = np.array(scores).flatten()
            scores = softmax(scores)

            msg = create_classification_message(self.classes, scores)

            transformation = output.getTransformation()
            if transformation is not None:
                msg.setTransformation(transformation)
            msg.setTimestamp(output.getTimestamp())
            msg.setSequenceNum(output.getSequenceNum())
            msg.setTimestampDevice(output.getTimestampDevice())

            self._logger.debug(f"Created message with {len(msg.classes)} classes")

            self.out.send(msg)

            self._logger.debug("Classification message sent successfully")
