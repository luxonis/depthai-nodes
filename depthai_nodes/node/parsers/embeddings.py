from typing import Any

import depthai as dai

from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.embeddings import compute_embeddings_output


class EmbeddingsParser(BaseParser):
    """Parser class for parsing the output of embeddings neural network model head.

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.

    Note:
        Emits ``dai.NNData`` messages. The output layer of the neural network model
        head.
    """

    def __init__(self) -> None:
        """Initialize the EmbeddingsParser node."""
        super().__init__()
        self.output_layer_name: str | None = None
        self._logger.debug(
            f"EmbeddingsParser initialized with output_layer_name={self.output_layer_name}"
        )

    def setOutputLayerNames(self, output_layer_name: str) -> None:
        """Sets the output layer name for the parser.

        Args:
            output_layer_name: The output layer name for the parser.
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")

        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to {self.output_layer_name}")

    def build(self, head_config: dict[str, Any]) -> "EmbeddingsParser":
        """Sets the head configuration for the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """
        output_names = self._normalize_output_layer_names(head_config["outputs"])
        assert (
            len(output_names) == 1
        ), "Embeddings head should have only one output layer"
        self.output_layer_name = output_names[0]

        self._logger.debug(
            f"EmbeddingsParser built with output_layer_name={self.output_layer_name}"
        )

        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("EmbeddingsParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped, no more data

            extracted = self.extract(output)
            computed = self.compute(extracted)
            self.emit(computed)

    def extract(self, output: dai.NNData) -> dai.NNData:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            The same NNData message, after checking that exactly one embedding output is
            selected.

        Raises:
            AssertionError: If the selected embedding output count is not one.
        """
        output_names = (
            [self.output_layer_name]
            if self.output_layer_name is not None
            else output.getAllLayerNames()
        )
        self._logger.debug(f"Processing input with layers: {output_names}")

        assert (
            len(output_names) == 1
        ), "Embeddings head should have only one output layer"
        return output

    @staticmethod
    def _normalize_output_layer_names(
        output_layer_name: str | list[str] | None,
    ) -> list[str]:
        if output_layer_name is None:
            return []
        if isinstance(output_layer_name, str):
            return [output_layer_name]
        return list(output_layer_name)

    @staticmethod
    def compute(output: dai.NNData) -> dai.NNData:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            output: Embedding payload, typically a ``dai.NNData`` message.

        Returns:
            The same object passed as ``output``; no copy or normalization is performed.

        Note:
            Uses
            `depthai_nodes.node.parsers.utils.embeddings.compute_embeddings_output`; see
            that helper for tensor layout and validation details.
        """
        return compute_embeddings_output(output)

    def emit(self, output: dai.NNData) -> None:
        """Forward the same NNData message on ``out``, preserving its metadata.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
        """
        output.setSequenceNum(output.getSequenceNum())
        output.setTimestamp(output.getTimestamp())
        output.setTimestampDevice(output.getTimestampDevice())
        transformation = output.getTransformation()
        if transformation is not None:
            output.setTransformation(transformation)

        self.out.send(output)
        self._logger.debug("Message sent successfully")
