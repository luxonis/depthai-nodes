from typing import Any, Dict

import depthai as dai

from .base_parser import BaseParser


class EmbeddingsParser(BaseParser):
    """Parser class for parsing the output of embeddings neural network model head.

    Attributes
    ----------

    Output Message/s
    ----------------
    **Type**: ImgDetectionsExtended

    **Description**: Message containing bounding boxes, labels, confidence scores, and keypoints or masks and protos of the detected objects.
    """

    def __init__(self) -> None:
        """Initialize the EmbeddingsParser node."""
        super().__init__()
        self.output_layer_names = None

    def build(self, head_config: Dict[str, Any]):
        """Sets the head configuration for the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.
        Returns
        -------
        EmbeddingsParser
            Returns the parser object with the head configuration set.
        """

        self.output_layer_names = head_config["outputs"]
        assert (
            len(self.output_layer_names) == 1
        ), "Embeddings head should have only one output layer"

        return self

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped, no more data
            # Get all the layer names
            output_names = self.output_layer_names or output.getAllLayerNames()
            assert (
                len(output_names) == 1
            ), "Embeddings head should have only one output layer"

            self.out.send(output)
