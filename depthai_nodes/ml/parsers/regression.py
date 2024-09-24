import depthai as dai
import numpy as np

from typing import Dict, List, Union

from ..messages.creators import create_regression_message
from .parser import Parser

class RegressionParser(Parser):
    """Parser class for parsing the output of a model with regression output (e.g. Age-Gender).

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    head_config : Dict
        Configuration of the head relevant to the parser.
    output_layer_name : str
        Name of the output layer relevant to the parser.
        
    Output Message/s
    ----------------
    **Type**: Predictions

    **Description**: Message containing the prediction(s).
    """

    def __init__(self):
        """Initializes the RegressionParser node.

        @param head_config: Configuration of the head relevant to the parser.
        @type head_config: Dict
        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name : str
        """

        super().__init__()
        self.output_layer_name: str = ""

    def build(
        self,
        heads: Union[List, Dict],
        head_name: str = "",
    ):
        super().build(heads, head_name)

        output_layers = self.head_config["outputs"]
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for Regression, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]

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

            predictions = output.getTensor(self.output_layer_name, dequantize=True).squeeze()
            predictions = np.atleast_1d(predictions).tolist()
            
            regression_message = create_regression_message(predictions=predictions)
            regression_message.setTimestamp(output.getTimestamp())

            self.out.send(regression_message)
