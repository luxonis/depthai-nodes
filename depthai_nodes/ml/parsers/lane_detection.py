from typing import Any, Dict, List, Tuple

import depthai as dai
import numpy as np

from ..messages.creators import create_cluster_message
from .base_parser import BaseParser
from .utils.ufld import decode_ufld


class LaneDetectionParser(BaseParser):
    """
    Parser class for Ultra-Fast-Lane-Detection model. It expects one ouput layer containing the lane detection results.
    It supports two versions of the model: CuLane and TuSimple. Results are representented with clusters of points.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    row_anchors : List[int]
        List of row anchors.
    griding_num : int
        Griding number.
    cls_num_per_lane : int
        Number of points per lane.
    input_shape : Tuple[int, int]
        Input shape.

    Output Message/s
    ----------------
    **Type**: Clusters
    **Description**: Detected lanes represented as clusters of points.

    Error Handling
    --------------
    **ValueError**: If the row anchors are not specified.
    **ValueError**: If the griding number is not specified.
    **ValueError**: If the number of points per lane is not specified.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        row_anchors: List[int] = None,
        griding_num: int = None,
        cls_num_per_lane: int = None,
        input_shape: Tuple[int, int] = (288, 800),
    ) -> None:
        """Initializes the lane detection parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param row_anchors: List of row anchors.
        @type row_anchors: List[int]
        @param griding_num: Griding number.
        @type griding_num: int
        @param cls_num_per_lane: Number of points per lane.
        @type cls_num_per_lane: int
        @param input_shape: Input shape.
        @type input_shape: Tuple[int, int]
        """
        super().__init__()
        self.output_layer_name = output_layer_name

        self.row_anchors = row_anchors
        self.griding_num = griding_num
        self.cls_num_per_lane = cls_num_per_lane
        self.input_shape = input_shape

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "LaneDetectionParser":
        """Configures the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.

        Returns
        -------
        LaneDetectionParser
            Returns the parser object with the head configuration set.
        """

        output_layers = head_config["outputs"]
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for LaneDetectionParser, got {len(output_layers)} layers."
            )
        self.output_layer_name = output_layers[0]
        self.row_anchors = head_config["row_anchors"]
        self.griding_num = head_config["griding_num"]
        self.cls_num_per_lane = head_config["cls_num_per_lane"]

        return self

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Set the output layer name for the lane detection model.

        @param output_layer_name: Name of the output layer.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name

    def setRowAnchors(self, row_anchors: List[int]) -> None:
        """Set the row anchors for the lane detection model.

        @param row_anchors: List of row anchors.
        @type row_anchors: List[int]
        """
        if not isinstance(row_anchors, list):
            raise ValueError("Row anchors must be a list.")
        if not all(isinstance(anchor, int) for anchor in row_anchors):
            raise ValueError("Row anchors must be a list of integers.")
        self.row_anchors = row_anchors

    def setGridingNum(self, griding_num: int) -> None:
        """Set the griding number for the lane detection model.

        @param griding_num: Griding number.
        @type griding_num: int
        """
        if not isinstance(griding_num, int):
            raise ValueError("Griding number must be an integer.")
        self.griding_num = griding_num

    def setClsNumPerLane(self, cls_num_per_lane: int) -> None:
        """Set the number of points per lane for the lane detection model.

        @param cls_num_per_lane: Number of classes per lane.
        @type cls_num_per_lane: int
        """
        if not isinstance(cls_num_per_lane, int):
            raise ValueError("Number of points per lane must be an integer.")
        self.cls_num_per_lane = cls_num_per_lane

    def setInputShape(self, input_shape: Tuple[int, int]) -> None:
        """Set the input shape for the lane detection model.

        @param input_shape: Input shape.
        @type input_shape: Tuple[int, int]
        """
        if not isinstance(input_shape, tuple):
            raise ValueError("Input shape must be a tuple.")
        if len(input_shape) != 2:
            raise ValueError("Input shape must be a tuple of two integers.")
        if not all(isinstance(size, int) for size in input_shape):
            raise ValueError("Input shape must be a tuple of integers.")
        self.input_shape = input_shape

    def run(self):
        if self.row_anchors is None:
            raise ValueError("Row anchors must be specified!")
        if self.griding_num is None:
            raise ValueError("Griding number must be specified!")
        if self.cls_num_per_lane is None:
            raise ValueError("Number of points per lane must be specified!")

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

            tensor = output.getTensor(self.output_layer_name, dequantize=True).astype(
                np.float32
            )
            y = tensor[0]

            points = decode_ufld(
                anchors=self.row_anchors,
                griding_num=self.griding_num,
                cls_num_per_lane=self.cls_num_per_lane,
                INPUT_WIDTH=self.input_shape[1],
                INPUT_HEIGHT=self.input_shape[0],
                y=y,
            )

            message = create_cluster_message(points)
            message.setTimestamp(output.getTimestamp())
            self.out.send(message)
