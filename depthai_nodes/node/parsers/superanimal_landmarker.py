from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_keypoints_message
from depthai_nodes.node.parsers.keypoints import KeypointParser
from depthai_nodes.node.parsers.utils.superanimal import (
    compute_superanimal_keypoints,
)


class SuperAnimalParser(KeypointParser):
    """Parser class for parsing the output of the SuperAnimal landmark model.

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.
        scale_factor (``float``): Scale factor to divide the keypoints by.
        n_keypoints (``int``): Number of keypoints.
        score_threshold (``float``): Confidence score threshold for detected keypoints.
        label_names (``list[str]``): Label names for the keypoints.
        edges (``list[tuple[int, int]]``): Pairs of keypoint indexes defining skeleton
            edges. For example, ``[(0, 1), (1, 2)]`` connects keypoint 0 to 1 and 1 to
            2.

    Note:
        Emits ``dai.beta.Keypoints`` messages. Output containing detected keypoints that
        exceed the confidence threshold.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        scale_factor: float = 256.0,
        n_keypoints: int = 39,
        score_threshold: float = 0.5,
        label_names: list[str] | None = None,
        edges: list[tuple[int, int]] | None = None,
    ) -> None:
        """Initializes the parser node.

        Args:
            output_layer_name: Name of the output layer relevant to the parser.
            n_keypoints: Number of keypoints.
            score_threshold: Confidence score threshold for detected keypoints.
            scale_factor: Scale factor to divide the keypoints by.
            label_names: Label names for the keypoints.
            edges: Pairs of keypoint indexes defining skeleton edges. For example,
                ``[(0, 1), (1, 2)]`` connects keypoint 0 to 1 and 1 to 2.
        """
        super().__init__(
            output_layer_name,
            scale_factor=scale_factor,
            n_keypoints=n_keypoints,
            score_threshold=score_threshold,
            label_names=label_names,
            edges=edges,
        )
        self._logger.debug(
            f"SuperAnimalParser initialized with output_layer_name='{output_layer_name}', scale_factor={scale_factor}, n_keypoints={n_keypoints}, score_threshold={score_threshold}, label_names={label_names}, edges={edges}"
        )

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "SuperAnimalParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        super().build(head_config)

        self._logger.debug("SuperAnimalParser built")

        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("SuperAnimalParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            heatmaps = self.extract(output)
            keypoints, scores = self.compute(heatmaps, scale_factor=self.scale_factor)
            self.emit(output, keypoints, scores)

    def extract(self, output: dai.NNData) -> np.ndarray:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Dequantized float32 heatmaps in the model tensor layout.

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
        heatmaps: np.ndarray,
        *,
        scale_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            heatmaps: Model heatmap tensor.
            scale_factor: Nonzero divisor used to convert model coordinates to
                normalized coordinates.

        Returns:
            A pair of ``(N, 2)`` keypoint coordinates and ``(N,)`` scores. Coordinates
            are divided by ``scale_factor``.

        Note:
            Uses
            `depthai_nodes.node.parsers.utils.superanimal.compute_superanimal_keypoints`;
            see that helper for tensor layout and validation details.
        """
        return compute_superanimal_keypoints(
            heatmaps,
            scale_factor=scale_factor,
        )

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
        msg = create_keypoints_message(
            keypoints,
            scores,
            self.score_threshold,
            label_names=self.label_names,
            edges=self.edges,
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
