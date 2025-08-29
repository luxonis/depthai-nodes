from typing import Any, Dict, List

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import (
    create_classification_message,
)
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils import softmax


class ClassificationParser(BaseParser):
    """Postprocessing logic for Classification model.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    classes : List[str]
        List of class names to be used for linking with their respective scores.
        Expected to be in the same order as Neural Network's output. If not provided, the message will only return sorted scores.
    is_softmax : bool = True
        If False, the scores are converted to probabilities using softmax function.

    Output Message/s
    ----------------
    **Type** : Classifications(dai.Buffer)

    **Description**: An object with attributes `classes` and `scores`. `classes` is a list of classes, sorted in descending order of scores. `scores` is a list of corresponding scores.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        classes: List[str] = None,
        is_softmax: bool = True,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param classes: List of class names to be used for linking with their respective
            scores. Expected to be in the same order as Neural Network's output. If not
            provided, the message will only return sorted scores.
        @type classes: List[str]
        @param is_softmax: If False, the scores are converted to probabilities using
            softmax function.
        @type is_softmax: bool
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.classes = classes or []
        self.n_classes = len(self.classes)
        self.is_softmax = is_softmax
        self._logger.debug(
            f"ClassificationParser initialized with output_layer_name='{output_layer_name}', classes={classes}, is_softmax={is_softmax}"
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

    def setSoftmax(self, is_softmax: bool) -> None:
        """Sets the softmax flag for the classification model.

        @param is_softmax: If False, the parser will convert the scores to probabilities
            using softmax function.
        @type is_softmax: bool
        """
        if not isinstance(is_softmax, bool):
            raise ValueError("is_softmax must be a boolean.")
        self.is_softmax = is_softmax
        self._logger.debug(f"Softmax set to {self.is_softmax}")

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "ClassificationParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: ClassificationParser
        """

        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for Classification, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.classes = head_config.get("classes", self.classes)
        self.n_classes = head_config.get("n_classes", self.n_classes)
        self.is_softmax = head_config.get("is_softmax", self.is_softmax)

        self._logger.debug(
            f"ClassificationParser built with output_layer_name='{self.output_layer_name}', classes={self.classes}, n_classes={self.n_classes}, is_softmax={self.is_softmax}"
        )
        return self

    def run(self):
        self._logger.debug("ClassificationParser run started")
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

            if len(scores) != self.n_classes and self.n_classes != 0:
                raise ValueError(
                    f"Number of labels and scores mismatch. Provided {self.n_classes} class names and {len(scores)} scores."
                )

            if not self.is_softmax:
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
