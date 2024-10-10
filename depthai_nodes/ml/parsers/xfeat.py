from typing import Any, Dict, Optional, Tuple

import depthai as dai
import numpy as np

from ..messages.creators import create_tracked_features_message
from .base_parser import BaseParser
from .utils.xfeat import detect_and_compute, match


class XFeatBaseParser(BaseParser):
    """Base parser class for parsing the output of the XFeat model. It is the parent
    class of the XFeatMonoParser and XFeatStereoParser classes.

    Attributes
    ----------
    input : Node.Input
        Node's input used in mono mode. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    reference_input : Node.Input
        Reference input for stereo mode. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    target_input : Node.Input
        Target input for stereo mode. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    output_layer_feats : str
        Name of the output layer containing features.
    output_layer_keypoints : str
        Name of the output layer containing keypoints.
    output_layer_heatmaps : str
        Name of the output layer containing heatmaps.
    original_size : Tuple[float, float]
        Original image size.
    input_size : Tuple[float, float]
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
        original_size: Tuple[float, float] = None,
        input_size: Tuple[float, float] = (640, 352),
        max_keypoints: int = 4096,
    ):
        """Initializes the XFeatBaseParser node."""
        super().__init__()
        self._target_input = self.createInput()  # used in stereo mode

        self.output_layer_feats = output_layer_feats
        self.output_layer_keypoints = output_layer_keypoints
        self.output_layer_heatmaps = output_layer_heatmaps
        self.original_size = original_size
        self.input_size = input_size
        self.max_keypoints = max_keypoints

    @property
    def reference_input(self) -> Optional[dai.Node.Input]:
        """Returns the reference input."""
        return self.input

    @property
    def target_input(self) -> Optional[dai.Node.Input]:
        """Returns the target input."""
        return self._target_input

    @reference_input.setter
    def reference_input(self, reference_input: Optional[dai.Node.Input]):
        """Sets the reference input."""
        self.input = reference_input

    @target_input.setter
    def target_input(self, target_input: Optional[dai.Node.Input]):
        """Sets the target input."""
        self._target_input = target_input

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
        XFeatBaseParser
            Returns the parser object with the head configuration set.
        """

        output_layers = head_config["outputs"]
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

        return self

    def setOutputLayerFeats(self, output_layer_feats):
        """Sets the output layer containing features.

        @param output_layer_feats: Name of the output layer containing features.
        @type output_layer_feats: str
        """
        self.output_layer_feats = output_layer_feats

    def setOutputLayerKeypoints(self, output_layer_keypoints):
        """Sets the output layer containing keypoints.

        @param output_layer_keypoints: Name of the output layer containing keypoints.
        @type output_layer_keypoints: str
        """
        self.output_layer_keypoints = output_layer_keypoints

    def setOutputLayerHeatmaps(self, output_layer_heatmaps):
        """Sets the output layer containing heatmaps.

        @param output_layer_heatmaps: Name of the output layer containing heatmaps.
        @type output_layer_heatmaps: str
        """
        self.output_layer_heatmaps = output_layer_heatmaps

    def setOriginalSize(self, original_size):
        """Sets the original image size.

        @param original_size: Original image size.
        @type original_size: Tuple[float, float]
        """
        self.original_size = original_size

    def setInputSize(self, input_size):
        """Sets the input image size.

        @param input_size: Input image size.
        @type input_size: Tuple[float, float]
        """
        self.input_size = input_size

    def setMaxKeypoints(self, max_keypoints):
        """Sets the maximum number of keypoints to keep.

        @param max_keypoints: Maximum number of keypoints.
        @type max_keypoints: int
        """
        self.max_keypoints = max_keypoints

    def validateParams(self):
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts the tensors from the output. It returns the features, keypoints, and
        heatmaps. It also handles the reshaping of the tensors.

        @param output: Output from the Neural Network node.
        @type output: dai.NNData
        @return: Tuple of features, keypoints, and heatmaps.
        @rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        feats = output.getTensor(self.output_layer_feats, dequantize=True).astype(
            np.float32
        )
        keypoints = output.getTensor(
            self.output_layer_keypoints, dequantize=True
        ).astype(np.float32)
        heatmaps = output.getTensor(self.output_layer_heatmaps, dequantize=True).astype(
            np.float32
        )

        if len(feats.shape) == 3:
            feats = feats.reshape((1,) + feats.shape).transpose(0, 3, 1, 2)
        if len(keypoints.shape) == 3:
            keypoints = keypoints.reshape((1,) + keypoints.shape).transpose(0, 3, 1, 2)
        if len(heatmaps.shape) == 3:
            heatmaps = heatmaps.reshape((1,) + heatmaps.shape).transpose(0, 3, 1, 2)

        return feats, keypoints, heatmaps


class XFeatMonoParser(XFeatBaseParser):
    """Parser class for parsing the output of the XFeat model. It can be used for
    parsing the output from one source (e.g. one camera). The reference frame can be set
    with trigger method.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    output_layer_feats : str
        Name of the output layer containing features.
    output_layer_keypoints : str
        Name of the output layer containing keypoints.
    output_layer_heatmaps : str
        Name of the output layer containing heatmaps.
    original_size : Tuple[float, float]
        Original image size.
    input_size : Tuple[float, float]
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
        output_layer_feats="feats",
        output_layer_keypoints="keypoints",
        output_layer_heatmaps="heatmaps",
        original_size: Tuple[float, float] = None,
        input_size: Tuple[float, float] = (640, 352),
        max_keypoints: int = 4096,
    ):
        """Initializes the XFeatParser node.

        @param output_layer_feats: Name of the output layer containing features.
        @type output_layer_feats: str
        @param output_layer_keypoints: Name of the output layer containing keypoints.
        @type output_layer_keypoints: str
        @param output_layer_heatmaps: Name of the output layer containing heatmaps.
        @type output_layer_heatmaps: str
        @param original_size: Original image size.
        @type original_size: Tuple[float, float]
        @param input_size: Input image size.
        @type input_size: Tuple[float, float]
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

    def setTrigger(self):
        """Sets the trigger to set the reference frame."""
        self.trigger = True

    def run(self):
        self.validateParams()

        resize_rate_w = self.original_size[0] / self.input_size[0]
        resize_rate_h = self.original_size[1] / self.input_size[1]

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            feats, keypoints, heatmaps = self.extractTensors(output)

            result = detect_and_compute(
                feats,
                keypoints,
                heatmaps,
                resize_rate_w,
                resize_rate_h,
                self.input_size,
                self.max_keypoints,
            )

            if result is not None:
                result = result[0]
            else:
                matched_points = dai.TrackedFeatures()
                matched_points.setTimestamp(output.getTimestamp())
                self.out.send(matched_points)
                continue

            if self.previous_results is not None:
                mkpts0, mkpts1 = match(self.previous_results, result)
                matched_points = create_tracked_features_message(mkpts0, mkpts1)
                matched_points.setTimestamp(output.getTimestamp())
                self.out.send(matched_points)
            else:
                matched_points = dai.TrackedFeatures()
                matched_points.setTimestamp(output.getTimestamp())
                self.out.send(matched_points)

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
    original_size : Tuple[float, float]
        Original image size.
    input_size : Tuple[float, float]
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
        output_layer_feats="feats",
        output_layer_keypoints="keypoints",
        output_layer_heatmaps="heatmaps",
        original_size: Tuple[float, float] = None,
        input_size: Tuple[float, float] = (640, 352),
        max_keypoints: int = 4096,
    ):
        """Initializes the XFeatParser node.

        @param output_layer_feats: Name of the output layer containing features.
        @type output_layer_feats: str
        @param output_layer_keypoints: Name of the output layer containing keypoints.
        @type output_layer_keypoints: str
        @param output_layer_heatmaps: Name of the output layer containing heatmaps.
        @type output_layer_heatmaps: str
        @param original_size: Original image size.
        @type original_size: Tuple[float, float]
        @param input_size: Input image size.
        @type input_size: Tuple[float, float]
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

    def run(self):
        self.validateParams()

        resize_rate_w = self.original_size[0] / self.input_size[0]
        resize_rate_h = self.original_size[1] / self.input_size[1]

        while self.isRunning():
            try:
                reference_output: dai.NNData = self.reference_input.get()
                target_output: dai.NNData = self.target_input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            (
                reference_feats,
                reference_keypoints,
                reference_heatmaps,
            ) = self.extractTensors(reference_output)
            target_feats, target_keypoints, target_heatmaps = self.extractTensors(
                target_output
            )

            reference_result = detect_and_compute(
                reference_feats,
                reference_keypoints,
                reference_heatmaps,
                resize_rate_w,
                resize_rate_h,
                self.input_size,
                self.max_keypoints,
            )

            target_result = detect_and_compute(
                target_feats,
                target_keypoints,
                target_heatmaps,
                resize_rate_w,
                resize_rate_h,
                self.input_size,
                self.max_keypoints,
            )

            if reference_result is not None:
                reference_result = reference_result[0]
            else:
                matched_points = dai.TrackedFeatures()
                matched_points.setTimestamp(reference_output.getTimestamp())
                self.out.send(matched_points)
                continue

            if target_result is not None:
                target_result = target_result[0]
            else:
                matched_points = dai.TrackedFeatures()
                matched_points.setTimestamp(target_output.getTimestamp())
                self.out.send(matched_points)
                continue

            mkpts0, mkpts1 = match(reference_result, target_result)
            matched_points = create_tracked_features_message(mkpts0, mkpts1)
            matched_points.setTimestamp(target_output.getTimestamp())
            self.out.send(matched_points)
