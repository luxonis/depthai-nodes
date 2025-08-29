from typing import Any, Dict

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_regression_message
from depthai_nodes.node.parsers.base_parser import BaseParser


class RegressionParser(BaseParser):
    """Parser class for parsing the output of a model with regression output (e.g. Age-
    Gender).

    Attributes
    ----------
    output_layer_name : str
        Name of the output layer relevant to the parser.

    Output Message/s
    ----------------
    **Type**: Predictions

    **Description**: Message containing the prediction(s).
    """

    def __init__(
        self,
        output_layer_name: str = "",
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name : str
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self._logger.debug(
            f"RegressionParser initialized with output_layer_name='{output_layer_name}'"
        )

    def setOutputLayerName(self, output_layer_name: str):
        """Sets the name of the output layer.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to '{self.output_layer_name}'")

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "RegressionParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: RegressionParser
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
        self._logger.debug("RegressionParser run started")
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

            predictions = output.getTensor(
                self.output_layer_name, dequantize=True
            ).squeeze()
            predictions = np.atleast_1d(predictions).tolist()

            regression_message = create_regression_message(predictions=predictions)
            regression_message.setTimestamp(output.getTimestamp())
            regression_message.setTimestampDevice(output.getTimestampDevice())
            regression_message.setSequenceNum(output.getSequenceNum())
            transformation = output.getTransformation()
            if transformation is not None:
                regression_message.setTransformation(transformation)

            self._logger.debug(
                f"Created regression message with {len(predictions)} values"
            )

            self.out.send(regression_message)

            self._logger.debug("Regression message sent successfully")
