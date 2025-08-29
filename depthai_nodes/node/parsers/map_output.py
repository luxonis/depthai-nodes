from typing import Any, Dict

import depthai as dai

from depthai_nodes.message.creators import create_map_message
from depthai_nodes.node.parsers.base_parser import BaseParser


class MapOutputParser(BaseParser):
    """A parser class for models that produce map outputs, such as depth maps (e.g.
    DepthAnything), density maps (e.g. DM-Count), heat maps, and similar.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    min_max_scaling : bool
        If True, the map is scaled to the range [0, 1].

    Output Message/s
    ----------------
    **Type**: Map2D

    **Description**: Density message containing the density map. The density map is represented with Map2D object.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        min_max_scaling: bool = False,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param min_max_scaling: If True, the map is scaled to the range [0, 1].
        @type min_max_scaling: bool
        """
        super().__init__()
        self.min_max_scaling = min_max_scaling
        self.output_layer_name = output_layer_name
        self._logger.debug(
            f"MapOutputParser initialized with output_layer_name='{output_layer_name}', min_max_scaling={min_max_scaling}"
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

    def setMinMaxScaling(self, min_max_scaling: bool) -> None:
        """Sets the min_max_scaling flag.

        @param min_max_scaling: If True, the map is scaled to the range [0, 1].
        @type min_max_scaling: bool
        """
        if not isinstance(min_max_scaling, bool):
            raise ValueError("min_max_scaling must be a boolean.")
        self.min_max_scaling = min_max_scaling
        self._logger.debug(f"Min max scaling set to {self.min_max_scaling}")

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "MapOutputParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: MapOutputParser
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
        self._logger.debug("MapOutputParser run started")
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

            map = output.getTensor(self.output_layer_name, dequantize=True)

            if map.shape[0] == 1:
                map = map[0]  # remove batch dimension

            map_message = create_map_message(
                map=map, min_max_scaling=self.min_max_scaling
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
