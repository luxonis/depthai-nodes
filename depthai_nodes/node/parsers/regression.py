from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_regression_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.regression import (
    compute_regression_predictions,
)


class RegressionParser(BaseParser):
    """Parser class for parsing the output of a model with regression output (e.g. Age-
    Gender).

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.

    Note:
        Emits ``dai.beta.Predictions`` messages. Message containing the prediction(s).
    """

    def __init__(
        self,
        output_layer_name: str = "",
    ) -> None:
        """Initializes the parser node.

        Args:
            output_layer_name: Name of the output layer relevant to the parser.
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self._logger.debug(
            f"RegressionParser initialized with output_layer_name='{output_layer_name}'"
        )

    def setOutputLayerName(self, output_layer_name: str):
        """Sets the name of the output layer.

        Args:
            output_layer_name: Name of the output layer relevant to the parser.
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to '{self.output_layer_name}'")

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "RegressionParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for Regression, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]

        self._logger.debug(
            f"RegressionParser built with output_layer_name='{self.output_layer_name}'"
        )

        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("RegressionParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            predictions = self.extract(output)
            predictions = self.compute(predictions)
            self.emit(output, predictions)

    def extract(self, output: dai.NNData) -> np.ndarray:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Dequantized regression tensor.

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
    def compute(predictions: np.ndarray) -> list[float]:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            predictions: Model prediction tensor.

        Returns:
            A Python list obtained after squeezing singleton dimensions. Scalar
            predictions become a one-item list; remaining non-singleton dimensions
            produce nested lists.

        Note:
            Uses
            `depthai_nodes.node.parsers.utils.regression.compute_regression_predictions`;
            see that helper for tensor layout and validation details.
        """
        return compute_regression_predictions(predictions)

    def emit(self, output: dai.NNData, predictions: list[float]) -> None:
        """Create a ``dai.beta.Predictions`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            predictions: Regression values returned by ``compute()``.
        """
        regression_message = create_regression_message(predictions=predictions)
        regression_message.setTimestamp(output.getTimestamp())
        regression_message.setTimestampDevice(output.getTimestampDevice())
        regression_message.setSequenceNum(output.getSequenceNum())
        transformation = output.getTransformation()
        if transformation is not None:
            regression_message.setTransformation(transformation)

        self._logger.debug(f"Created regression message with {len(predictions)} values")
        self.out.send(regression_message)
        self._logger.debug("Regression message sent successfully")
