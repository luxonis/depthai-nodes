from typing import Any, Dict

import depthai as dai
import numpy as np

from ..messages.creators import create_regression_message
from .base_parser import BaseParser


class RegressionParser(BaseParser):
    """Parser class for parsing the output of a model with regression output (e.g. Age-
    Gender).

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
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
    ):
        """Initializes the RegressionParser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name : str
        """
        super().__init__()
        self.output_layer_name = output_layer_name

    def build(
        self,
        head_config: Dict[str, Any],
    ):
        """Sets the head configuration for the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.

        Returns
        -------
        RegressionParser
            Returns the parser object with the head configuration set.
        """
        output_layers = head_config["outputs"]
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for Regression, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]

        return self

    def setOutputLayerName(self, output_layer_name: str):
        """Sets the name of the output layer.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        """
        self.output_layer_name = output_layer_name

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

            predictions = output.getTensor(
                self.output_layer_name, dequantize=True
            ).squeeze()
            predictions = np.atleast_1d(predictions).tolist()

            regression_message = create_regression_message(predictions=predictions)
            regression_message.setTimestamp(output.getTimestamp())

            self.out.send(regression_message)
