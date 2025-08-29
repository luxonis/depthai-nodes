from typing import Any, Dict, List, Optional, Tuple

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_keypoints_message
from depthai_nodes.node.parsers.base_parser import BaseParser


class KeypointParser(BaseParser):
    """Parser class for 2D or 3D keypoints models. It expects one ouput layer containing
    keypoints. The number of keypoints must be specified. Moreover, the keypoints are
    normalized by a scale factor if provided.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    scale_factor : float
        Scale factor to divide the keypoints by.
    n_keypoints : int
        Number of keypoints the model detects.
    score_threshold : float
        Confidence score threshold for detected keypoints.
    label_names : List[str]
        Label names for the keypoints.
    edges : List[Tuple[int, int]]
        Keypoint connection pairs for visualizing the skeleton. Example: [(0,1), (1,2), (2,3), (3,0)] shows that keypoint 0 is connected to keypoint 1, keypoint 1 is connected to keypoint 2, etc.

    Output Message/s
    ----------------
    **Type**: Keypoints

    **Description**: Keypoints message containing 2D or 3D keypoints.

    Error Handling
    --------------
    **ValueError**: If the number of keypoints is not specified.

    **ValueError**: If the number of coordinates per keypoint is not 2 or 3.

    **ValueError**: If the number of output layers is not 1.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        scale_factor: float = 1.0,
        n_keypoints: int = None,
        score_threshold: float = None,
        label_names: Optional[List[str]] = None,
        edges: Optional[List[List[int]]] = None,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        @param n_keypoints: Number of keypoints.
        @type n_keypoints: int
        @param label_names: Label names for the keypoints.
        @type label_names: Optional[List[str]]
        @param edges: Keypoint connection pairs for visualizing the skeleton. Example:
            [(0,1), (1,2), (2,3), (3,0)] shows that keypoint 0 is connected to keypoint
            1, keypoint 1 is connected to keypoint 2, etc.
        @type edges: Optional[List[Tuple[int, int]]]
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

        @param output_layer_name: The name of the output layer.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to '{self.output_layer_name}'")

    def setScaleFactor(self, scale_factor: float) -> None:
        """Sets the scale factor to divide the keypoints by.

        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        """
        if not isinstance(scale_factor, float):
            raise ValueError("Scale factor must be a float.")

        if scale_factor <= 0:
            raise ValueError("Scale factor must be greater than 0.")

        self.scale_factor = scale_factor
        self._logger.debug(f"Scale factor set to {self.scale_factor}")

    def setNumKeypoints(self, n_keypoints: int) -> None:
        """Sets the number of keypoints.

        @param n_keypoints: Number of keypoints.
        @type n_keypoints: int
        """
        if not isinstance(n_keypoints, int):
            raise ValueError("Number of keypoints must be an integer.")

        if n_keypoints <= 0:
            raise ValueError("Number of keypoints must be greater than 0.")

        self.n_keypoints = n_keypoints
        self._logger.debug(f"Number of keypoints set to {self.n_keypoints}")

    def setScoreThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for the detected body keypoints.

        @param threshold: Confidence score threshold for detected keypoints.
        @type threshold: float
        """
        if not isinstance(threshold, float):
            raise ValueError("Confidence threshold must be a float.")

        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")

        self.score_threshold = threshold
        self._logger.debug(f"Score threshold set to {self.score_threshold}")

    def setLabelNames(self, label_names: List[str]) -> None:
        """Sets the label names for the keypoints.

        @param label_names: List of label names for the keypoints.
        @type label_names: List[str]
        """
        if not isinstance(label_names, list):
            raise ValueError("Label names must be a list.")
        if not all(isinstance(label, str) for label in label_names):
            raise ValueError("Label names must be a list of strings.")
        self.label_names = label_names
        self._logger.debug(f"Label names set to {self.label_names}")

    def setEdges(self, edges: List[Tuple[int, int]]) -> None:
        """Sets the edges for the keypoints.

        @param edges: List of edges for the keypoints. Example: [(0,1), (1,2), (2,3),
            (3,0)] shows that keypoint 0 is connected to keypoint 1, keypoint 1 is
            connected to keypoint 2, etc.
        @type edges: List[Tuple[int, int]]
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
        head_config: Dict[str, Any],
    ) -> "KeypointParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: KeypointParser
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
        self._logger.debug("KeypointParser run started")
        if self.n_keypoints is None:
            raise ValueError("Number of keypoints must be specified!")

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

            keypoints = output.getTensor(
                self.output_layer_name, dequantize=True
            ).astype(np.float32)
            num_coords = int(np.prod(keypoints.shape) / self.n_keypoints)

            if num_coords not in [2, 3]:
                raise ValueError(
                    f"Expected 2 or 3 coordinates per keypoint, got {num_coords}."
                )

            keypoints = keypoints.reshape(self.n_keypoints, num_coords)

            keypoints /= self.scale_factor

            keypoints = np.clip(keypoints, 0, 1)

            msg = create_keypoints_message(
                keypoints, edges=self.edges, label_names=self.label_names
            )
            msg.setTimestamp(output.getTimestamp())
            msg.setSequenceNum(output.getSequenceNum())
            msg.setTimestampDevice(output.getTimestampDevice())
            transformation = output.getTransformation()
            if transformation is not None:
                msg.setTransformation(transformation)

            self._logger.debug(
                f"Created keypoints message with {len(keypoints)} points"
            )

            self.out.send(msg)

            self._logger.debug("Keypoints message sent successfully")
