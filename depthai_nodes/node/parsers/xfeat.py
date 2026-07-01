from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_tracked_features_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.xfeat import (
    compute_xfeat_matches,
    compute_xfeat_result,
)


class XFeatBaseParser(BaseParser):
    """Base parser class for parsing the output of the XFeat model. It is the parent
    class of the XFeatMonoParser and XFeatStereoParser classes.

    Attributes
    ----------
    reference_input : Node.Input
        Reference input for stereo mode. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    target_input : Node.Input
        Target input for stereo mode. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    output_layer_feats : str
        Name of the output layer containing features.
    output_layer_keypoints : str
        Name of the output layer containing keypoints.
    output_layer_heatmaps : str
        Name of the output layer containing heatmaps.
    original_size : tuple[float, float]
        Original image size.
    input_size : tuple[float, float]
        Input image size.
    max_keypoints : int
        Maximum number of keypoints to keep.

    Error Handling
    --------------
    **ValueError**: If the number of output layers is not E{3}.
    **ValueError**: If the original image size is not specified.
    **ValueError**: If the input image size is not specified.
    **ValueError**: If the maximum number of keypoints is not specified.
    **ValueError**: If the output layer containing features is not specified.
    **ValueError**: If the output layer containing keypoints is not specified.
    **ValueError**: If the output layer containing heatmaps is not specified.
    """

    def __init__(
        self,
        output_layer_feats: str = "",
        output_layer_keypoints: str = "",
        output_layer_heatmaps: str = "",
        original_size: tuple[float, float] = None,
        input_size: tuple[float, float] = (640, 352),
        max_keypoints: int = 4096,
    ) -> None:
        """Initializes the parser node."""
        super().__init__()
        self._target_input = self.createInput()  # used in stereo mode

        self.output_layer_feats = output_layer_feats
        self.output_layer_keypoints = output_layer_keypoints
        self.output_layer_heatmaps = output_layer_heatmaps
        self.original_size = original_size
        self.input_size = input_size
        self.max_keypoints = max_keypoints

        self._logger.debug(
            f"XFeatBaseParser initialized with output_layer_feats='{output_layer_feats}', output_layer_keypoints='{output_layer_keypoints}', output_layer_heatmaps='{output_layer_heatmaps}', original_size={original_size}, input_size={input_size}, max_keypoints={max_keypoints}"
        )

    @property
    def reference_input(self) -> dai.Node.Input | None:
        """Returns the reference input."""
        return self.input

    @property
    def target_input(self) -> dai.Node.Input | None:
        """Returns the target input."""
        return self._target_input

    @reference_input.setter
    def reference_input(self, reference_input: dai.Node.Input | None):
        """Sets the reference input."""
        self.input = reference_input

    @target_input.setter
    def target_input(self, target_input: dai.Node.Input | None):
        """Sets the target input."""
        self._target_input = target_input

    def setOutputLayerFeats(self, output_layer_feats: str) -> None:
        """Sets the output layer containing features.

        @param output_layer_feats: Name of the output layer containing features.
        @type output_layer_feats: str
        """
        if not isinstance(output_layer_feats, str):
            raise ValueError("Output layer containing features must be a string!")
        self.output_layer_feats = output_layer_feats
        self._logger.debug(
            f"Output layer containing features set to '{self.output_layer_feats}'"
        )

    def setOutputLayerKeypoints(self, output_layer_keypoints: str) -> None:
        """Sets the output layer containing keypoints.

        @param output_layer_keypoints: Name of the output layer containing keypoints.
        @type output_layer_keypoints: str
        """
        if not isinstance(output_layer_keypoints, str):
            raise ValueError("Output layer containing keypoints must be a string!")
        self.output_layer_keypoints = output_layer_keypoints
        self._logger.debug(
            f"Output layer containing keypoints set to '{self.output_layer_keypoints}'"
        )

    def setOutputLayerHeatmaps(self, output_layer_heatmaps: str) -> None:
        """Sets the output layer containing heatmaps.

        @param output_layer_heatmaps: Name of the output layer containing heatmaps.
        @type output_layer_heatmaps: str
        """
        if not isinstance(output_layer_heatmaps, str):
            raise ValueError("Output layer containing heatmaps must be a string!")
        self.output_layer_heatmaps = output_layer_heatmaps
        self._logger.debug(
            f"Output layer containing heatmaps set to '{self.output_layer_heatmaps}'"
        )

    def setOriginalSize(self, original_size: tuple[int, int]) -> None:
        """Sets the original image size.

        @param original_size: Original image size.
        @type original_size: tuple[int, int]
        """
        if not isinstance(original_size, tuple) or len(original_size) != 2:
            raise ValueError("Original image size must be a tuple of two ints!")
        for size in original_size:
            if not isinstance(size, int):
                raise ValueError("Original image size must be a tuple of two ints!")
        self.original_size = original_size
        self._logger.debug(f"Original image size set to {self.original_size}")

    def setInputSize(self, input_size: tuple[int, int]) -> None:
        """Sets the input image size.

        @param input_size: Input image size.
        @type input_size: tuple[int, int]
        """
        if not isinstance(input_size, tuple) or len(input_size) != 2:
            raise ValueError("Input image size must be a tuple of two ints!")
        for size in input_size:
            if not isinstance(size, int):
                raise ValueError("Input image size must be a tuple of two ints!")
        self.input_size = input_size
        self._logger.debug(f"Input image size set to {self.input_size}")

    def setMaxKeypoints(self, max_keypoints: int) -> None:
        """Sets the maximum number of keypoints to keep.

        @param max_keypoints: Maximum number of keypoints.
        @type max_keypoints: int
        """
        if not isinstance(max_keypoints, int):
            raise ValueError("Maximum number of keypoints must be an int!")
        self.max_keypoints = max_keypoints
        self._logger.debug(f"Maximum number of keypoints set to {self.max_keypoints}")

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "XFeatBaseParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: XFeatBaseParser
        """

        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 3:
            raise ValueError(
                f"Only three output layers supported for XFeat, got {len(output_layers)} layers."
            )

        for layer in output_layers:
            if "feats" in layer:
                self.output_layer_feats = layer
            elif "keypoints" in layer:
                self.output_layer_keypoints = layer
            elif "heatmaps" in layer:
                self.output_layer_heatmaps = layer

        self.original_size = head_config.get("original_size", self.original_size)
        self.input_size = head_config.get("input_size", self.input_size)
        self.max_keypoints = head_config.get("max_keypoints", self.max_keypoints)

        self._logger.debug(
            f"XFeatBaseParser built with output_layer_feats='{self.output_layer_feats}', output_layer_keypoints='{self.output_layer_keypoints}', output_layer_heatmaps='{self.output_layer_heatmaps}', original_size={self.original_size}, input_size={self.input_size}, max_keypoints={self.max_keypoints}"
        )

        return self

    def validateParams(self) -> None:
        """Validates the parameters."""
        if self.original_size is None:
            raise ValueError("Original image size must be specified!")
        if self.input_size is None:
            raise ValueError("Input image size must be specified!")
        if self.max_keypoints is None:
            raise ValueError("Maximum number of keypoints must be specified!")
        if self.output_layer_feats == "":
            raise ValueError("Output layer containing features must be specified!")
        if self.output_layer_keypoints == "":
            raise ValueError("Output layer containing keypoints must be specified!")
        if self.output_layer_heatmaps == "":
            raise ValueError("Output layer containing heatmaps must be specified!")

    def extractTensors(
        self, output: dai.NNData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts the tensors from the output. It returns the features, keypoints, and
        heatmaps. It also handles the reshaping of the tensors by requesting the NCHW
        storage order.

        @param output: Output from the Neural Network node.
        @type output: dai.NNData
        @return: Tuple of features, keypoints, and heatmaps.
        @rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        feats = output.getTensor(
            self.output_layer_feats,
            dequantize=True,
            storageOrder=dai.TensorInfo.StorageOrder.NCHW,
        ).astype(np.float32)
        keypoints = output.getTensor(
            self.output_layer_keypoints,
            dequantize=True,
            storageOrder=dai.TensorInfo.StorageOrder.NCHW,
        ).astype(np.float32)
        heatmaps = output.getTensor(
            self.output_layer_heatmaps,
            dequantize=True,
            storageOrder=dai.TensorInfo.StorageOrder.NCHW,
        ).astype(np.float32)

        return feats, keypoints, heatmaps


class XFeatMonoParser(XFeatBaseParser):
    """Parser class for parsing the output of the XFeat model. It can be used for
    parsing the output from one source (e.g. one camera). The reference frame can be set
    with trigger method.

    Attributes
    ----------
    output_layer_feats : str
        Name of the output layer containing features.
    output_layer_keypoints : str
        Name of the output layer containing keypoints.
    output_layer_heatmaps : str
        Name of the output layer containing heatmaps.
    original_size : tuple[float, float]
        Original image size.
    input_size : tuple[float, float]
        Input image size.
    max_keypoints : int
        Maximum number of keypoints to keep.
    previous_results : np.ndarray
        Previous results from the model. Previous results are used to match keypoints between two frames.
    trigger : bool
        Trigger to set the reference frame.

    Output Message/s
    ----------------
    **Type**: dai.TrackedFeatures

    **Description**: TrackedFeatures message containing matched keypoints with the same ID.

    Error Handling
    --------------
    **ValueError**: If the original image size is not specified.
    **ValueError**: If the input image size is not specified.
    **ValueError**: If the maximum number of keypoints is not specified.
    **ValueError**: If the output layer containing features is not specified.
    **ValueError**: If the output layer containing keypoints is not specified.
    **ValueError**: If the output layer containing heatmaps is not specified.
    """

    def __init__(
        self,
        output_layer_feats: str = "feats",
        output_layer_keypoints: str = "keypoints",
        output_layer_heatmaps: str = "heatmaps",
        original_size: tuple[float, float] = None,
        input_size: tuple[float, float] = (640, 352),
        max_keypoints: int = 4096,
    ) -> None:
        """Initializes the XFeatParser node.

        @param output_layer_feats: Name of the output layer containing features.
        @type output_layer_feats: str
        @param output_layer_keypoints: Name of the output layer containing keypoints.
        @type output_layer_keypoints: str
        @param output_layer_heatmaps: Name of the output layer containing heatmaps.
        @type output_layer_heatmaps: str
        @param original_size: Original image size.
        @type original_size: tuple[float, float]
        @param input_size: Input image size.
        @type input_size: tuple[float, float]
        @param max_keypoints: Maximum number of keypoints to keep.
        @type max_keypoints: int
        """
        super().__init__(
            output_layer_feats,
            output_layer_keypoints,
            output_layer_heatmaps,
            original_size,
            input_size,
            max_keypoints,
        )

        self.previous_results = None
        self.trigger = False

        self._logger.debug(
            f"XFeatMonoParser initialized with output_layer_feats='{output_layer_feats}', output_layer_keypoints='{output_layer_keypoints}', output_layer_heatmaps='{output_layer_heatmaps}', original_size={original_size}, input_size={input_size}, max_keypoints={max_keypoints}"
        )

    def setTrigger(self) -> None:
        """Sets the trigger to set the reference frame."""
        self.trigger = True
        self._logger.debug(f"Trigger set to {self.trigger}")

    def run(self):
        self._logger.debug("XFeatMonoParser run started")
        self.validateParams()

        resize_rate_w = self.original_size[0] / self.input_size[0]
        resize_rate_h = self.original_size[1] / self.input_size[1]

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            feats, keypoints, heatmaps = self.extract(output)
            result = self.compute(
                feats,
                keypoints,
                heatmaps,
                resize_rate_w=resize_rate_w,
                resize_rate_h=resize_rate_h,
            )
            self.emit(output, result)

    def extract(self, output: dai.NNData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._logger.debug(f"Processing input with layers: {output.getAllLayerNames()}")
        return self.extractTensors(output)

    def compute(
        self,
        feats: np.ndarray,
        keypoints: np.ndarray,
        heatmaps: np.ndarray,
        *,
        resize_rate_w: float,
        resize_rate_h: float,
    ) -> dict[str, Any] | None:
        result = compute_xfeat_result(
            feats,
            keypoints,
            heatmaps,
            resize_rate_w=resize_rate_w,
            resize_rate_h=resize_rate_h,
            input_size=self.input_size,
            max_keypoints=self.max_keypoints,
        )
        if result is not None:
            return result[0]
        return None

    def emit(self, output: dai.NNData, result: dict[str, Any] | None) -> None:
        if result is None:
            matched_points = dai.TrackedFeatures()
            matched_points.setTimestamp(output.getTimestamp())
            matched_points.setSequenceNum(output.getSequenceNum())
            self._logger.debug("No keypoints found, sending TrackedFeatures message")
            self.out.send(matched_points)
            self._logger.debug("TrackedFeatures message sent")
            return

        if self.previous_results is not None:
            mkpts0, mkpts1 = compute_xfeat_matches(self.previous_results, result)
            matched_points = create_tracked_features_message(mkpts0, mkpts1)
            matched_points.setTimestamp(output.getTimestamp())
            matched_points.setSequenceNum(output.getSequenceNum())
            self._logger.debug("Keypoints found, sending TrackedFeatures message")
            self.out.send(matched_points)
            self._logger.debug("TrackedFeatures message sent")
        else:
            matched_points = dai.TrackedFeatures()
            matched_points.setTimestamp(output.getTimestamp())
            matched_points.setSequenceNum(output.getSequenceNum())
            self._logger.debug("No previous results, sending TrackedFeatures message")
            self.out.send(matched_points)
            self._logger.debug("TrackedFeatures message sent")

        if self.trigger:
            self.previous_results = result
            self.trigger = False


class XFeatStereoParser(XFeatBaseParser):
    """Parser class for parsing the output of the XFeat model. It can be used for parsing the output from two sources (e.g. two cameras - left and right).

    Attributes
    ----------
    reference_input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    target_input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    output_layer_feats : str
        Name of the output layer from which the features are extracted.
    output_layer_keypoints : str
        Name of the output layer from which the keypoints are extracted.
    output_layer_heatmaps : str
        Name of the output layer from which the heatmaps are extracted.
    original_size : tuple[float, float]
        Original image size.
    input_size : tuple[float, float]
        Input image size.
    max_keypoints : int
        Maximum number of keypoints to keep.

    Output Message/s
    ----------------
    **Type**: dai.TrackedFeatures

    **Description**: TrackedFeatures message containing matched keypoints with the same ID.

    Error Handling
    --------------
    **ValueError**: If the original image size is not specified.
    **ValueError**: If the input image size is not specified.
    **ValueError**: If the maximum number of keypoints is not specified.
    **ValueError**: If the output layer containing features is not specified.
    **ValueError**: If the output layer containing keypoints is not specified.
    **ValueError**: If the output layer containing heatmaps is not specified.
    """

    def __init__(
        self,
        output_layer_feats: str = "feats",
        output_layer_keypoints: str = "keypoints",
        output_layer_heatmaps: str = "heatmaps",
        original_size: tuple[float, float] = None,
        input_size: tuple[float, float] = (640, 352),
        max_keypoints: int = 4096,
    ) -> None:
        """Initializes the XFeatParser node.

        @param output_layer_feats: Name of the output layer containing features.
        @type output_layer_feats: str
        @param output_layer_keypoints: Name of the output layer containing keypoints.
        @type output_layer_keypoints: str
        @param output_layer_heatmaps: Name of the output layer containing heatmaps.
        @type output_layer_heatmaps: str
        @param original_size: Original image size.
        @type original_size: tuple[float, float]
        @param input_size: Input image size.
        @type input_size: tuple[float, float]
        @param max_keypoints: Maximum number of keypoints to keep.
        @type max_keypoints: int
        """
        super().__init__(
            output_layer_feats,
            output_layer_keypoints,
            output_layer_heatmaps,
            original_size,
            input_size,
            max_keypoints,
        )

        self._logger.debug(
            f"XFeatStereoParser initialized with output_layer_feats='{output_layer_feats}', output_layer_keypoints='{output_layer_keypoints}', output_layer_heatmaps='{output_layer_heatmaps}', original_size={original_size}, input_size={input_size}, max_keypoints={max_keypoints}"
        )

    def run(self):
        self._logger.debug("XFeatStereoParser run started")
        self.validateParams()

        resize_rate_w = self.original_size[0] / self.input_size[0]
        resize_rate_h = self.original_size[1] / self.input_size[1]

        while self.isRunning():
            try:
                reference_output: dai.NNData = self.reference_input.get()
                target_output: dai.NNData = self.target_input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            reference_tensors, target_tensors = self.extract(
                reference_output, target_output
            )
            match_result = self.compute(
                reference_tensors,
                target_tensors,
                resize_rate_w=resize_rate_w,
                resize_rate_h=resize_rate_h,
            )
            self.emit(reference_output, target_output, match_result)

    def extract(
        self, reference_output: dai.NNData, target_output: dai.NNData
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        self._logger.debug(
            f"Processing reference input with layers: {reference_output.getAllLayerNames()}"
        )
        self._logger.debug(
            f"Processing target input with layers: {target_output.getAllLayerNames()}"
        )
        return self.extractTensors(reference_output), self.extractTensors(target_output)

    def compute(
        self,
        reference_tensors: tuple[np.ndarray, np.ndarray, np.ndarray],
        target_tensors: tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        resize_rate_w: float,
        resize_rate_h: float,
    ) -> dict[str, tuple[np.ndarray, np.ndarray] | str | None]:
        reference_result = compute_xfeat_result(
            *reference_tensors,
            resize_rate_w=resize_rate_w,
            resize_rate_h=resize_rate_h,
            input_size=self.input_size,
            max_keypoints=self.max_keypoints,
        )
        target_result = compute_xfeat_result(
            *target_tensors,
            resize_rate_w=resize_rate_w,
            resize_rate_h=resize_rate_h,
            input_size=self.input_size,
            max_keypoints=self.max_keypoints,
        )

        if reference_result is None:
            return {"status": "reference_missing", "match_result": None}
        if target_result is None:
            return {"status": "target_missing", "match_result": None}

        return {
            "status": "matched",
            "match_result": compute_xfeat_matches(
                reference_result[0], target_result[0]
            ),
        }

    def emit(
        self,
        reference_output: dai.NNData,
        target_output: dai.NNData,
        result: dict[str, tuple[np.ndarray, np.ndarray] | str | None],
    ) -> None:
        status = result["status"]
        match_result = result["match_result"]
        if match_result is None:
            matched_points = dai.TrackedFeatures()
            if status == "reference_missing":
                matched_points.setTimestamp(reference_output.getTimestamp())
                self._logger.debug(
                    "No reference keypoints found, sending TrackedFeatures message"
                )
            else:
                matched_points.setTimestamp(target_output.getTimestamp())
                self._logger.debug(
                    "No target keypoints found, sending TrackedFeatures message"
                )
            matched_points.setSequenceNum(reference_output.getSequenceNum())
            self.out.send(matched_points)
            self._logger.debug("TrackedFeatures message sent")
            return

        mkpts0, mkpts1 = match_result
        matched_points = create_tracked_features_message(mkpts0, mkpts1)
        matched_points.setTimestamp(target_output.getTimestamp())
        matched_points.setSequenceNum(reference_output.getSequenceNum())
        matched_points.setTimestampDevice(target_output.getTimestampDevice())
        self._logger.debug("Keypoints found, sending TrackedFeatures message")
        self.out.send(matched_points)
        self._logger.debug("TrackedFeatures message sent")
