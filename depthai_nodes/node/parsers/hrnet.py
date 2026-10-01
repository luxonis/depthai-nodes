from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_keypoints_message
from depthai_nodes.node.parsers.keypoints import KeypointParser
from depthai_nodes.node.parsers.utils.hrnet import compute_hrnet_keypoints


class HRNetParser(KeypointParser):
    """Parser class for parsing the output of the HRNet pose estimation model. The code is
    inspired by https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation.

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.
        score_threshold (``float``): Confidence score threshold for detected keypoints.
        label_names (``list[str] | None``): Label names for the keypoints.
        edges (``list[tuple[int, int]] | None``): Pairs of keypoint indexes defining
            skeleton edges. For example, ``[(0, 1), (1, 2)]`` connects keypoint 0 to 1
            and 1 to 2.

    Note:
        Emits ``dai.beta.Keypoints`` messages. Output containing detected body
        keypoints.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        score_threshold: float = 0.5,
        label_names: list[str] | None = None,
        edges: list[tuple[int, int]] | None = None,
    ) -> None:
        """Initializes the parser node.

        Args:
            output_layer_name: Name of the output layer relevant to the parser.
            score_threshold: Confidence score threshold for detected keypoints.
            label_names: Label names for the keypoints.
            edges: Pairs of keypoint indexes defining skeleton edges. For example,
                ``[(0, 1), (1, 2)]`` connects keypoint 0 to 1 and 1 to 2.
        """
        super().__init__(
            output_layer_name,
            score_threshold=score_threshold,
            label_names=label_names,
            edges=edges,
        )
        self._logger.debug(
            f"HRNetParser initialized with output_layer_name='{output_layer_name}', score_threshold={score_threshold}, label_names={label_names}, edges={edges}"
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

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "HRNetParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        super().build(head_config)
        self._logger.debug("HRNetParser built successfully")
        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("HRNetParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            heatmaps = self.extract(output)
            keypoints, scores = self.compute(heatmaps)
            self.emit(output, keypoints, scores)

    def extract(self, output: dai.NNData) -> np.ndarray:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Dequantized heatmaps requested in NCHW storage order.

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

        return output.getTensor(
            self.output_layer_name,
            dai.TensorInfo.StorageOrder.NCHW,
            dequantize=True,
        )

    @staticmethod
    def compute(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            heatmaps: Heatmaps of shape ``(1, keypoints, height, width)`` or
                ``(keypoints, height, width)``. The first axis is removed when its size
                is 1.

        Returns:
            A pair of normalized ``(N, 2)`` XY coordinates and ``(N,)`` peak scores
            clipped to [0, 1].

        Note:
            Uses `depthai_nodes.node.parsers.utils.hrnet.compute_hrnet_keypoints`; see
            that helper for tensor layout and validation details.
        """
        keypoints, scores = compute_hrnet_keypoints(heatmaps)
        return keypoints, scores

    def emit(
        self, output: dai.NNData, keypoints: np.ndarray, scores: np.ndarray
    ) -> None:
        """Create a ``dai.beta.Keypoints`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            keypoints: Normalized keypoint coordinates returned by ``compute()``.
            scores: Confidence scores corresponding to the computed payload.
        """
        keypoints_message = create_keypoints_message(
            keypoints=keypoints,
            scores=scores,
            confidence_threshold=self.score_threshold,
            edges=self.edges,
            label_names=self.label_names,
        )
        keypoints_message.setTimestamp(output.getTimestamp())
        keypoints_message.setSequenceNum(output.getSequenceNum())
        keypoints_message.setTimestampDevice(output.getTimestampDevice())
        transformation = output.getTransformation()
        if transformation is not None:
            keypoints_message.setTransformation(transformation)

        self._logger.debug(f"Created keypoints message with {len(keypoints)} points")
        self.out.send(keypoints_message)
        self._logger.debug("Keypoint output sent successfully")
