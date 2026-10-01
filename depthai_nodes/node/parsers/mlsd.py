from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_line_detection_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.mlsd import compute_mlsd_lines


class MLSDParser(BaseParser):
    """Parser class for parsing the output of the M-LSD line detection model. The parser
    is specifically designed to parse the output of the M-LSD model. As the result, the
    node sends out the detected lines in the form of a message.

    Attributes:
        output_layer_tpmap (``str``): Name of the output layer containing the tpMap
            tensor.
        output_layer_heat (``str``): Name of the output layer containing the heat
            tensor.
        topk_n (``int``): Number of top candidates to keep.
        score_thr (``float``): Confidence score threshold for detected lines.
        dist_thr (``float``): Minimum line length in output-map pixels.

    Note:
        Emits ``dai.beta.Lines`` messages. Native message containing detected lines and
        confidence scores.
    """

    def __init__(
        self,
        output_layer_tpmap: str = "",
        output_layer_heat: str = "",
        topk_n: int = 200,
        score_thr: float = 0.10,
        dist_thr: float = 20.0,
    ) -> None:
        """Initializes the parser node.

        Args:
            topk_n: Number of top candidates to keep.
            score_thr: Confidence score threshold for detected lines.
            dist_thr: Minimum line length in output-map pixels.
            output_layer_tpmap: Name of the output tensor containing line displacements.
            output_layer_heat: Name of the output tensor containing the heatmap.
        """
        super().__init__()
        self.output_layer_tpmap = output_layer_tpmap
        self.output_layer_heat = output_layer_heat

        self.topk_n = topk_n
        self.score_thr = score_thr
        self.dist_thr = dist_thr
        self._logger.debug(
            f"MLSDParser initialized with output_layer_tpmap='{output_layer_tpmap}', output_layer_heat='{output_layer_heat}', topk_n={topk_n}, score_thr={score_thr}, dist_thr={dist_thr}"
        )

    def setOutputLayerTPMap(self, output_layer_tpmap: str) -> None:
        """Sets the name of the output layer containing the tpMap tensor.

        Args:
            output_layer_tpmap: Name of the output layer containing the tpMap tensor.
        """
        if not isinstance(output_layer_tpmap, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_tpmap = output_layer_tpmap
        self._logger.debug(f"Output layer tpmap set to '{self.output_layer_tpmap}'")

    def setOutputLayerHeat(self, output_layer_heat: str) -> None:
        """Sets the name of the output layer containing the heat tensor.

        Args:
            output_layer_heat: Name of the output layer containing the heat tensor.
        """
        if not isinstance(output_layer_heat, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_heat = output_layer_heat
        self._logger.debug(f"Output layer heat set to '{self.output_layer_heat}'")

    def setTopK(self, topk_n: int) -> None:
        """Sets the number of top candidates to keep.

        Args:
            topk_n: Number of top candidates to keep.
        """
        if not isinstance(topk_n, int):
            raise ValueError("topk_n must be an integer.")
        self.topk_n = topk_n
        self._logger.debug(f"Topk_n set to {self.topk_n}")

    def setScoreThreshold(self, score_thr: float) -> None:
        """Sets the confidence score threshold for detected lines.

        Args:
            score_thr: Confidence score threshold for detected lines.
        """
        if not isinstance(score_thr, float):
            raise ValueError("score_thr must be a float.")
        self.score_thr = score_thr
        self._logger.debug(f"Score threshold set to {self.score_thr}")

    def setDistanceThreshold(self, dist_thr: float) -> None:
        """Sets the distance threshold for merging lines.

        Args:
            dist_thr: Minimum line length in output-map pixels.
        """
        if not isinstance(dist_thr, float):
            raise ValueError("dist_thr must be a float.")
        self.dist_thr = dist_thr
        self._logger.debug(f"Distance threshold set to {self.dist_thr}")

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "MLSDParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 2:
            raise ValueError(
                f"Only two output layers are supported for MLSDParser, got {len(output_layers)} layers."
            )
        for layer in output_layers:
            if "tpMap" in layer:
                self.output_layer_tpmap = layer
            elif "heat" in layer:
                self.output_layer_heat = layer
        self.topk_n = head_config.get("topk_n", self.topk_n)
        self.score_thr = head_config.get("score_thr", self.score_thr)
        self.dist_thr = head_config.get("dist_thr", self.dist_thr)

        self._logger.debug(
            f"MLSDParser built with output_layer_tpmap='{self.output_layer_tpmap}', output_layer_heat='{self.output_layer_heat}', topk_n={self.topk_n}, score_thr={self.score_thr}, dist_thr={self.dist_thr}"
        )

        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("MLSDParser run started")
        if self.output_layer_tpmap == "":
            raise ValueError(
                "Output layer containing the tpMap tensor is not set. Please use setOutputLayerTPMap method or correct NN archive."
            )
        if self.output_layer_heat == "":
            raise ValueError(
                "Output layer containing the heat tensor is not set. Please use setOutputLayerHeat method or correct NN archive."
            )

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            tpMap, heat_np = self.extract(output)
            lines, scores = self.compute(
                tpMap,
                heat_np,
                topk_n=self.topk_n,
                score_thr=self.score_thr,
                dist_thr=self.dist_thr,
            )
            self.emit(output, lines, scores)

    def extract(self, output: dai.NNData) -> tuple[np.ndarray, np.ndarray]:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Float32 displacement and heat tensors. Displacement is requested in NCHW
            order.
        """
        self._logger.debug(f"Processing input with layers: {output.getAllLayerNames()}")
        tpMap = output.getTensor(
            self.output_layer_tpmap,
            dequantize=True,
            storageOrder=dai.TensorInfo.StorageOrder.NCHW,
        ).astype(np.float32)
        heat_np = output.getTensor(self.output_layer_heat, dequantize=True).astype(
            np.float32
        )
        return tpMap, heat_np

    @staticmethod
    def compute(
        tpMap: np.ndarray,
        heat_np: np.ndarray,
        *,
        topk_n: int,
        score_thr: float,
        dist_thr: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            tpMap: Four-dimensional line-displacement tensor in NCHW layout.
            heat_np: Heat tensor used to rank line-center candidates.
            topk_n: Maximum number of line-center candidates to examine.
            score_thr: Minimum candidate score.
            dist_thr: Minimum line length in output-map pixels.

        Returns:
            Normalized endpoint coordinates of shape ``(N, 4)`` and float32 line scores.

        Note:
            Uses `depthai_nodes.node.parsers.utils.mlsd.compute_mlsd_lines`; see that
            helper for tensor layout and validation details.
        """
        return compute_mlsd_lines(
            tpMap,
            heat_np,
            topk_n=topk_n,
            score_thr=score_thr,
            dist_thr=dist_thr,
        )

    def emit(self, output: dai.NNData, lines: np.ndarray, scores: np.ndarray) -> None:
        """Create a ``dai.beta.Lines`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            lines: Normalized ``[x1, y1, x2, y2]`` line endpoints.
            scores: Confidence scores corresponding to the computed payload.
        """
        message = create_line_detection_message(lines, scores)
        message.setTimestamp(output.getTimestamp())
        message.setSequenceNum(output.getSequenceNum())
        message.setTimestampDevice(output.getTimestampDevice())
        transformation = output.getTransformation()
        if transformation is not None:
            message.setTransformation(transformation)

        self._logger.debug(f"Created line detection message with {len(lines)} lines")
        self.out.send(message)
        self._logger.debug("Line detection message sent successfully")
