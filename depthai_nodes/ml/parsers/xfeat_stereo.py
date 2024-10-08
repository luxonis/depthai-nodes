from typing import Tuple

import depthai as dai
import numpy as np

from ..messages.creators import create_tracked_features_message
from .utils.xfeat import detect_and_compute, match


class XFeatStereoParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the XFeat model. It can be used for parsing the output from two sources (e.g. two cameras - left and right).

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
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
    """

    def __init__(
        self,
        original_size: Tuple[float, float] = None,
        input_size: Tuple[float, float] = (640, 352),
        max_keypoints: int = 4096,
    ):
        """Initializes the XFeatParser node.

        @param original_size: Original image size.
        @type original_size: Tuple[float, float]
        @param input_size: Input image size.
        @type input_size: Tuple[float, float]
        @param max_keypoints: Maximum number of keypoints to keep.
        @type max_keypoints: int
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.reference_input = self.createInput()
        self.target_input = self.createInput()
        self.out = self.createOutput()
        self.original_size = original_size
        self.input_size = input_size
        self.max_keypoints = max_keypoints

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

    def run(self):
        if self.original_size is None:
            raise ValueError("Original image size must be specified!")

        resize_rate_w = self.original_size[0] / self.input_size[0]
        resize_rate_h = self.original_size[1] / self.input_size[1]

        while self.isRunning():
            try:
                reference_output: dai.NNData = self.reference_input.get()
                target_output: dai.NNData = self.target_input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            reference_feats = reference_output.getTensor(
                "feats", dequantize=True
            ).astype(np.float32)
            reference_keypoints = reference_output.getTensor(
                "keypoints", dequantize=True
            ).astype(np.float32)
            reference_heatmaps = reference_output.getTensor(
                "heatmaps", dequantize=True
            ).astype(np.float32)

            target_feats = target_output.getTensor("feats", dequantize=True).astype(
                np.float32
            )
            target_keypoints = target_output.getTensor(
                "keypoints", dequantize=True
            ).astype(np.float32)
            target_heatmaps = target_output.getTensor(
                "heatmaps", dequantize=True
            ).astype(np.float32)

            if len(reference_feats.shape) == 3:
                reference_feats = reference_feats.reshape(
                    (1,) + reference_feats.shape
                ).transpose(0, 3, 1, 2)
            if len(reference_keypoints.shape) == 3:
                reference_keypoints = reference_keypoints.reshape(
                    (1,) + reference_keypoints.shape
                ).transpose(0, 3, 1, 2)
            if len(reference_heatmaps.shape) == 3:
                reference_heatmaps = reference_heatmaps.reshape(
                    (1,) + reference_heatmaps.shape
                ).transpose(0, 3, 1, 2)

            if len(target_feats.shape) == 3:
                target_feats = target_feats.reshape(
                    (1,) + target_feats.shape
                ).transpose(0, 3, 1, 2)
            if len(target_keypoints.shape) == 3:
                target_keypoints = target_keypoints.reshape(
                    (1,) + target_keypoints.shape
                ).transpose(0, 3, 1, 2)
            if len(target_heatmaps.shape) == 3:
                target_heatmaps = target_heatmaps.reshape(
                    (1,) + target_heatmaps.shape
                ).transpose(0, 3, 1, 2)

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
