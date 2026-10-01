from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_cluster_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.lane_detection import (
    compute_lane_detection_points,
)


class LaneDetectionParser(BaseParser):
    """Parser class for Ultra-Fast-Lane-Detection model. It expects one output layer
    containing the lane detection results. It supports two versions of the model: CuLane
    and TuSimple. Results are representented with clusters of points.

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.
        row_anchors (``list[int]``): List of row anchors.
        griding_num (``int``): Griding number.
        cls_num_per_lane (``int``): Number of points per lane.
        input_size (``tuple[int, int]``): Input size (width,height).

    Output messages:

    **Type**: dai.beta.Clusters
    **Description**: Detected lanes represented as clusters of points.

    Raises:
        ValueError: If the row anchors are not specified.
        ValueError: If the griding number is not specified.
        ValueError: If the number of points per lane is not specified.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        row_anchors: list[int] = None,
        griding_num: int = None,
        cls_num_per_lane: int = None,
        input_size: tuple[int, int] = None,
    ) -> None:
        """Initializes the lane detection parser node.

        Args:
            output_layer_name: Name of the output layer relevant to the parser.
            row_anchors: List of row anchors.
            griding_num: Griding number.
            cls_num_per_lane: Number of points per lane.
            input_size: Input size (width,height).
        """
        super().__init__()
        self.output_layer_name = output_layer_name

        self.row_anchors = row_anchors
        self.griding_num = griding_num
        self.cls_num_per_lane = cls_num_per_lane
        self.input_size = input_size
        self._logger.debug(
            f"LaneDetectionParser initialized with output_layer_name='{output_layer_name}', row_anchors={row_anchors}, griding_num={griding_num}, cls_num_per_lane={cls_num_per_lane}, input_size={input_size}"
        )

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Set the output layer name for the lane detection model.

        Args:
            output_layer_name: Name of the output layer.
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to '{self.output_layer_name}'")

    def setRowAnchors(self, row_anchors: list[int]) -> None:
        """Set the row anchors for the lane detection model.

        Args:
            row_anchors: List of row anchors.
        """
        if not isinstance(row_anchors, list):
            raise ValueError("Row anchors must be a list.")
        if not all(isinstance(anchor, int) for anchor in row_anchors):
            raise ValueError("Row anchors must be a list of integers.")
        self.row_anchors = row_anchors
        self._logger.debug(f"Row anchors set to {self.row_anchors}")

    def setGridingNum(self, griding_num: int) -> None:
        """Set the griding number for the lane detection model.

        Args:
            griding_num: Griding number.
        """
        if not isinstance(griding_num, int):
            raise ValueError("Griding number must be an integer.")
        self.griding_num = griding_num
        self._logger.debug(f"Griding number set to {self.griding_num}")

    def setClsNumPerLane(self, cls_num_per_lane: int) -> None:
        """Set the number of points per lane for the lane detection model.

        Args:
            cls_num_per_lane: Number of classes per lane.
        """
        if not isinstance(cls_num_per_lane, int):
            raise ValueError("Number of points per lane must be an integer.")
        self.cls_num_per_lane = cls_num_per_lane
        self._logger.debug(f"Number of points per lane set to {self.cls_num_per_lane}")

    def setInputSize(self, input_size: tuple[int, int]) -> None:
        """Set the input size for the lane detection model.

        Args:
            input_size: Input size (width,height).
        """
        if not isinstance(input_size, tuple):
            raise ValueError("Input size must be a tuple.")
        if len(input_size) != 2:
            raise ValueError("Input size must be a tuple of two integers.")
        if not all(isinstance(size, int) for size in input_size):
            raise ValueError("Input size must be a tuple of integers.")
        self.input_size = input_size
        self._logger.debug(f"Input size set to {self.input_size}")

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "LaneDetectionParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for LaneDetectionParser, got {len(output_layers)} layers."
            )
        self.output_layer_name = output_layers[0]
        self.row_anchors = head_config.get("row_anchors", self.row_anchors)
        self.griding_num = head_config.get("griding_num", self.griding_num)
        self.cls_num_per_lane = head_config.get(
            "cls_num_per_lane", self.cls_num_per_lane
        )

        inputs = head_config["model_inputs"]
        if len(inputs) != 1:
            raise ValueError(
                f"Only one input supported for LaneDetectionParser, got {len(inputs)} inputs."
            )
        self.input_shape = inputs[0].get("shape")
        self.layout = inputs[0].get("layout")
        if self.layout == "NHWC":
            self.input_size = (self.input_shape[2], self.input_shape[1])
        elif self.layout == "NCHW":
            self.input_size = (self.input_shape[3], self.input_shape[2])
        else:
            raise ValueError(
                f"Input layout {self.layout} not supported for input_size extraction."
            )

        self._logger.debug(
            f"LaneDetectionParser built with output_layer_name='{self.output_layer_name}', row_anchors={self.row_anchors}, griding_num={self.griding_num}, cls_num_per_lane={self.cls_num_per_lane}, input_size={self.input_size}"
        )

        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("LaneDetectionParser run started")
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

            tensor = self.extract(output)
            points = self.compute(
                tensor,
                row_anchors=self.row_anchors,
                griding_num=self.griding_num,
                cls_num_per_lane=self.cls_num_per_lane,
                input_size=self.input_size,
            )
            self.emit(output, points)

    def extract(self, output: dai.NNData) -> np.ndarray:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Dequantized float32 batched UFLD grid tensor.

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

        return output.getTensor(self.output_layer_name, dequantize=True).astype(
            np.float32
        )

    @staticmethod
    def compute(
        tensor: np.ndarray,
        *,
        row_anchors: list[int],
        griding_num: int,
        cls_num_per_lane: int,
        input_size: tuple[int, int],
    ) -> list[list[tuple[int, int]]]:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            tensor: Batched logits with grid classes, sampled rows, and lanes as the
                remaining axes; only the first batch item is used.
            row_anchors: Image row positions, in pixels, for the lane sampling grid.
            griding_num: Number of horizontal grid cells, excluding the no-lane class.
            cls_num_per_lane: Number of sampled row positions per lane.
            input_size: Model input size as ``(width, height)``.

        Returns:
            One list per lane containing normalized XY point tuples. Lanes with fewer
            than three valid samples have empty lists.

        Note:
            Uses
            `depthai_nodes.node.parsers.utils.lane_detection.compute_lane_detection_points`;
            see that helper for tensor layout and validation details.
        """
        return compute_lane_detection_points(
            tensor,
            row_anchors=row_anchors,
            griding_num=griding_num,
            cls_num_per_lane=cls_num_per_lane,
            input_size=input_size,
        )

    def emit(self, output: dai.NNData, points) -> None:
        """Create a ``dai.beta.Clusters`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            points: Lists of normalized XY points, one list per lane.
        """
        msg = create_cluster_message(points)
        msg.setTimestamp(output.getTimestamp())
        msg.setSequenceNum(output.getSequenceNum())
        msg.setTimestampDevice(output.getTimestampDevice())
        transformation = output.getTransformation()
        if transformation is not None:
            msg.setTransformation(transformation)

        self._logger.debug(f"Created lane detection message with {len(points)} points")
        self.out.send(msg)
        self._logger.debug("Lane detection message sent successfully")
