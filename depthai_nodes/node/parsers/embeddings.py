from typing import Any, Dict

import depthai as dai

from depthai_nodes.node.parsers.base_parser import BaseParser


class EmbeddingsParser(BaseParser):
    """Parser class for parsing the output of embeddings neural network model head.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.

    Output Message/s
    ----------------
    **Type**: dai.NNData

    **Description**: The output layer of the neural network model head.
    """

    def __init__(self) -> None:
        """Initialize the EmbeddingsParser node."""
        super().__init__()
        self.output_layer_name: str = None
        self._logger.debug(
            f"EmbeddingsParser initialized with output_layer_name={self.output_layer_name}"
        )

    def setOutputLayerNames(self, output_layer_name: str) -> None:
        """Sets the output layer name for the parser.

        @param output_layer_name: The output layer name for the parser.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")

        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to {self.output_layer_name}")

    def build(self, head_config: Dict[str, Any]) -> "EmbeddingsParser":
        """Sets the head configuration for the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: EmbeddingsParser
        """

        self.output_layer_name = head_config["outputs"]
        assert (
            len(self.output_layer_name) == 1
        ), "Embeddings head should have only one output layer"

        self._logger.debug(
            f"EmbeddingsParser built with output_layer_name={self.output_layer_name}"
        )

        return self

    def run(self):
        self._logger.debug("EmbeddingsParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped, no more data

            # Get all the layer names
            output_names = self.output_layer_name or output.getAllLayerNames()
            self._logger.debug(f"Processing input with layers: {output_names}")

            assert (
                len(output_names) == 1
            ), "Embeddings head should have only one output layer"
            output.setSequenceNum(output.getSequenceNum())
            output.setTimestamp(output.getTimestamp())
            output.setTimestampDevice(output.getTimestampDevice())
            transformation = output.getTransformation()
            if transformation is not None:
                output.setTransformation(transformation)

            self.out.send(output)

            self._logger.debug("Message sent successfully")
