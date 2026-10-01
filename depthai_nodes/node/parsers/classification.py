from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import (
    create_classification_message,
)
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils import compute_classification_scores


class ClassificationParser(BaseParser):
    """Postprocessing logic for Classification model.

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.
        classes (``list[str]``): List of class names to be used for linking with their
            respective scores. Expected to be in the same order as Neural Network's
            output. If not provided, the message will only return sorted scores.
        is_softmax (``bool = True``): If False, the scores are converted to
            probabilities using softmax function.

    Output messages:

    **Type** : dai.beta.Classifications

    **Description**: An object with attributes ``classes`` and ``scores``. ``classes``
    is a list of classes, sorted in descending order of scores. ``scores`` is a list of
    corresponding scores.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        classes: list[str] = None,
        is_softmax: bool = True,
    ) -> None:
        """Initializes the parser node.

        Args:
            output_layer_name: Name of the output layer relevant to the parser.
            classes: List of class names to be used for linking with their respective
                scores. Expected to be in the same order as Neural Network's output. If
                not provided, the message will only return sorted scores.
            is_softmax: If False, the scores are converted to probabilities using
                softmax function.
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

        Args:
            output_layer_name: The name of the output layer.
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to '{self.output_layer_name}'")

    def setClasses(self, classes: list[str]) -> None:
        """Sets the class names for the classification model.

        Args:
            classes: List of class names to be used for linking with their respective
                scores.
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

        Args:
            is_softmax: If False, the parser will convert the scores to probabilities
                using softmax function.
        """
        if not isinstance(is_softmax, bool):
            raise ValueError("is_softmax must be a boolean.")
        self.is_softmax = is_softmax
        self._logger.debug(f"Softmax set to {self.is_softmax}")

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "ClassificationParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
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
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("ClassificationParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break

            scores = self.extract(output)
            scores = self.compute(scores, is_softmax=self.is_softmax)
            self.emit(output, scores)

    def extract(self, output: dai.NNData) -> np.ndarray:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Flattened dequantized class scores. Their count must match configured
            classes when a nonzero class count is set.

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

        scores = output.getTensor(self.output_layer_name, dequantize=True).flatten()

        if len(scores) != self.n_classes and self.n_classes != 0:
            raise ValueError(
                f"Number of labels and scores mismatch. Provided {self.n_classes} class names and {len(scores)} scores."
            )

        return scores

    @staticmethod
    def compute(
        scores: np.ndarray,
        *,
        is_softmax: bool = True,
    ) -> np.ndarray:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            scores: Model score tensor.
            is_softmax: Whether scores already contain probabilities. If false, apply
                softmax.

        Returns:
            One-dimensional array with one score per class.

        Note:
            Uses
            `depthai_nodes.node.parsers.utils.classification.compute_classification_scores`;
            see that helper for tensor layout and validation details.
        """
        return compute_classification_scores(scores, is_softmax=is_softmax)

    def emit(self, output: dai.NNData, scores: np.ndarray) -> None:
        """Create a ``dai.beta.Classifications`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            scores: Confidence scores corresponding to the computed payload.
        """
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
