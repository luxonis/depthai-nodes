import depthai as dai
from typing import Any, Dict

from ..messages.creators import create_map_message

from .base_parser import BaseParser


class MapOutputParser(BaseParser):
    """A parser class for models that produce map outputs, such as depth maps (e.g.
    DepthAnything), density maps (e.g. DM-Count), heat maps, and similar.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
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
        """Initializes the MapOutputParser node."""
        super().__init__()
        self.min_max_scaling = min_max_scaling
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
        MapOutputParser
            Returns the parser object with the head configuration set.
        """

        output_layers = head_config["outputs"]
        if len(output_layers) != 1:
            raise ValueError(
                f"MapOutputParser expects exactly 1 output layers, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.min_max_scaling = head_config.get("min_max_scaling", self.min_max_scaling)

        return self

    def setMinMaxScaling(self, min_max_scaling: bool):
        """Sets the min_max_scaling flag.

        @param min_max_scaling: If True, the map is scaled to the range [0, 1].
        @type min_max_scaling: bool
        """
        self.min_max_scaling = min_max_scaling

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            map = output.getTensor(self.output_layer_name, dequantize=True)

            if map.shape[0] == 1:
                map = map[0]  # remove batch dimension

            map_message = create_map_message(
                map=map, min_max_scaling=self.min_max_scaling
            )

            self.out.send(map_message)
