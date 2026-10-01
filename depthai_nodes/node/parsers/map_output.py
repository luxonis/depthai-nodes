from typing import Any

import depthai as dai

from depthai_nodes.message.creators import create_map_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.map_output import compute_map_output


class MapOutputParser(BaseParser):
    """A parser class for models that produce map outputs, such as depth maps (e.g.
    DepthAnything), density maps (e.g. DM-Count), heat maps, and similar.

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.
        min_max_scaling (``bool``): If True, the map is scaled to the range [0, 1].

    Note:
        Emits ``dai.beta.Map2D`` messages. Map2D message containing the parsed map as a
        native dai.beta.Map2D object.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        min_max_scaling: bool = False,
    ) -> None:
        """Initializes the parser node.

        Args:
            output_layer_name: Name of the output layer relevant to the parser.
            min_max_scaling: If True, the map is scaled to the range [0, 1].
        """
        super().__init__()
        self.min_max_scaling = min_max_scaling
        self.output_layer_name = output_layer_name
        self._logger.debug(
            f"MapOutputParser initialized with output_layer_name='{output_layer_name}', min_max_scaling={min_max_scaling}"
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

    def setMinMaxScaling(self, min_max_scaling: bool) -> None:
        """Sets the min_max_scaling flag.

        Args:
            min_max_scaling: If True, the map is scaled to the range [0, 1].
        """
        if not isinstance(min_max_scaling, bool):
            raise ValueError("min_max_scaling must be a boolean.")
        self.min_max_scaling = min_max_scaling
        self._logger.debug(f"Min max scaling set to {self.min_max_scaling}")

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "MapOutputParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 1:
            raise ValueError(
                f"MapOutputParser expects exactly 1 output layers, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.min_max_scaling = head_config.get("min_max_scaling", self.min_max_scaling)

        self._logger.debug(
            f"MapOutputParser built with output_layer_name='{self.output_layer_name}', min_max_scaling={self.min_max_scaling}"
        )

        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("MapOutputParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            map_tensor = self.extract(output)
            map_output = self.compute(map_tensor)
            self.emit(output, map_output)

    def extract(self, output: dai.NNData):
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Dequantized numeric map tensor.

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
    def compute(map_tensor):
        """Compute parser results from extracted tensors without sending messages.

        Args:
            map_tensor: HW map, a map with leading singleton axes, or an HW1 map.

        Returns:
            A two-dimensional array. Values and dtype are preserved; the result may
            share input storage.

        Note:
            Uses `depthai_nodes.node.parsers.utils.map_output.compute_map_output`; see
            that helper for tensor layout and validation details.
        """
        return compute_map_output(map_tensor)

    def emit(self, output: dai.NNData, map_output) -> None:
        """Create a ``dai.beta.Map2D`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            map_output: Two-dimensional map returned by ``compute()``.
        """
        map_message = create_map_message(
            map_array=map_output, min_max_scaling=self.min_max_scaling
        )
        map_message.setTimestamp(output.getTimestamp())
        map_message.setTimestampDevice(output.getTimestampDevice())
        map_message.setSequenceNum(output.getSequenceNum())
        transformation = output.getTransformation()
        if transformation is not None:
            map_message.setTransformation(transformation)

        self._logger.debug("Created Map message.")
        self.out.send(map_message)
        self._logger.debug("Map message sent successfully")
