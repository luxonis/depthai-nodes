from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_keypoints_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.keypoints import compute_keypoints


class KeypointParser(BaseParser):
    """Parser class for 2D or 3D keypoints models. It expects one output layer
    containing keypoints. The number of keypoints must be specified. Moreover, the
    keypoints are normalized by a scale factor if provided.

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.
        scale_factor (``float``): Scale factor to divide the keypoints by.
        n_keypoints (``int``): Number of keypoints the model detects.
        score_threshold (``float``): Confidence score threshold for detected keypoints.
        label_names (``list[str]``): Label names for the keypoints.
        edges (``list[tuple[int, int]]``): Pairs of keypoint indexes defining skeleton
            edges. For example, ``[(0, 1), (1, 2)]`` connects keypoint 0 to 1 and 1 to
            2.

    Note:
        Emits ``dai.beta.Keypoints`` messages. Output containing 2D or 3D keypoints.

    Raises:
        ValueError: If the number of keypoints is not specified.

        ValueError: If the number of coordinates per keypoint is not 2 or 3.

        ValueError: If the number of output layers is not 1.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        scale_factor: float = 1.0,
        n_keypoints: int = None,
        score_threshold: float = None,
        label_names: list[str] | None = None,
        edges: list[list[int]] | None = None,
    ) -> None:
        """Initializes the parser node.

        Args:
            output_layer_name: Name of the output layer relevant to the parser.
            scale_factor: Scale factor to divide the keypoints by.
            n_keypoints: Number of keypoints.
            label_names: Label names for the keypoints.
            edges: Pairs of keypoint indexes defining skeleton edges. For example,
                ``[(0, 1), (1, 2)]`` connects keypoint 0 to 1 and 1 to 2.
            score_threshold: Optional confidence threshold stored in the parser
                configuration.
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.scale_factor = scale_factor
        self.n_keypoints = n_keypoints
        self.score_threshold = score_threshold
        self.label_names = label_names
        self.edges = edges
        self._logger.debug(
            f"KeypointParser initialized with output_layer_name='{output_layer_name}', scale_factor={scale_factor}, n_keypoints={n_keypoints}, score_threshold={score_threshold}, label_names={label_names}, edges={edges}"
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

    def setScaleFactor(self, scale_factor: float) -> None:
        """Sets the scale factor to divide the keypoints by.

        Args:
            scale_factor: Scale factor to divide the keypoints by.
        """
        if not isinstance(scale_factor, float):
            raise ValueError("Scale factor must be a float.")

        if scale_factor <= 0:
            raise ValueError("Scale factor must be greater than 0.")

        self.scale_factor = scale_factor
        self._logger.debug(f"Scale factor set to {self.scale_factor}")

    def setNumKeypoints(self, n_keypoints: int) -> None:
        """Sets the number of keypoints.

        Args:
            n_keypoints: Number of keypoints.
        """
        if not isinstance(n_keypoints, int):
            raise ValueError("Number of keypoints must be an integer.")

        if n_keypoints <= 0:
            raise ValueError("Number of keypoints must be greater than 0.")

        self.n_keypoints = n_keypoints
        self._logger.debug(f"Number of keypoints set to {self.n_keypoints}")

    def setScoreThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for the detected body keypoints.

        Args:
            threshold: Confidence score threshold for detected keypoints.
        """
        if not isinstance(threshold, float):
            raise ValueError("Confidence threshold must be a float.")

        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")

        self.score_threshold = threshold
        self._logger.debug(f"Score threshold set to {self.score_threshold}")

    def setLabelNames(self, label_names: list[str]) -> None:
        """Sets the label names for the keypoints.

        Args:
            label_names: List of label names for the keypoints.
        """
        if not isinstance(label_names, list):
            raise ValueError("Label names must be a list.")
        if not all(isinstance(label, str) for label in label_names):
            raise ValueError("Label names must be a list of strings.")
        self.label_names = label_names
        self._logger.debug(f"Label names set to {self.label_names}")

    def setEdges(self, edges: list[tuple[int, int]]) -> None:
        """Sets the edges for the keypoints.

        Args:
            edges: List of edges for the keypoints. Example: [(0,1), (1,2), (2,3),
                (3,0)] shows that keypoint 0 is connected to keypoint 1, keypoint 1 is
                connected to keypoint 2, etc.
        """
        if not isinstance(edges, list):
            raise ValueError("Edges must be a list.")
        if not all(
            isinstance(edge, tuple)
            and len(edge) == 2
            and all(isinstance(i, int) for i in edge)
            for edge in edges
        ):
            raise ValueError("Edges must be a list of tuples of integers.")
        self.edges = edges
        self._logger.debug(f"Edges set to {self.edges}")

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "KeypointParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        output_layers = head_config["outputs"]
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for Keypoint, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.scale_factor = head_config.get("scale_factor", self.scale_factor)
        self.n_keypoints = head_config.get("n_keypoints", self.n_keypoints)
        self.score_threshold = head_config.get("score_threshold", self.score_threshold)
        self.label_names = head_config.get("keypoint_labels", self.label_names)
        keypoint_edges = head_config.get("skeleton_edges", self.edges)
        if keypoint_edges:
            self.edges = [tuple(edge) for edge in keypoint_edges]

        self._logger.debug(
            f"KeypointParser built with output_layer_name='{self.output_layer_name}', scale_factor={self.scale_factor}, n_keypoints={self.n_keypoints}, score_threshold={self.score_threshold}, label_names={self.label_names}, edges={self.edges}"
        )

        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("KeypointParser run started")
        if self.n_keypoints is None:
            raise ValueError("Number of keypoints must be specified!")

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            keypoints = self.extract(output)
            keypoints = self.compute(
                keypoints,
                n_keypoints=self.n_keypoints,
                scale_factor=self.scale_factor,
            )
            self.emit(output, keypoints)

    def extract(self, output: dai.NNData) -> np.ndarray:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Dequantized float32 keypoint tensor.

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
        keypoints: np.ndarray,
        *,
        n_keypoints: int,
        scale_factor: float = 1.0,
    ) -> np.ndarray:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            keypoints: Model keypoint tensor.
            n_keypoints: Number of keypoints encoded per prediction.
            scale_factor: Nonzero divisor used to convert model coordinates to
                normalized coordinates.

        Returns:
            Float32 coordinates of shape ``(n_keypoints, 2)`` or ``(n_keypoints, 3)``,
            divided by ``scale_factor`` and clipped to [0, 1].

        Note:
            Uses `depthai_nodes.node.parsers.utils.keypoints.compute_keypoints`; see
            that helper for tensor layout and validation details.
        """
        return compute_keypoints(
            keypoints,
            n_keypoints=n_keypoints,
            scale_factor=scale_factor,
        )

    def emit(self, output: dai.NNData, keypoints: np.ndarray) -> None:
        """Create a ``dai.beta.Keypoints`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            keypoints: Normalized keypoint coordinates returned by ``compute()``.
        """
        msg = create_keypoints_message(
            keypoints, edges=self.edges, label_names=self.label_names
        )
        msg.setTimestamp(output.getTimestamp())
        msg.setSequenceNum(output.getSequenceNum())
        msg.setTimestampDevice(output.getTimestampDevice())
        transformation = output.getTransformation()
        if transformation is not None:
            msg.setTransformation(transformation)

        self._logger.debug(f"Created keypoints message with {len(keypoints)} points")
        self.out.send(msg)
        self._logger.debug("Keypoint output sent successfully")
