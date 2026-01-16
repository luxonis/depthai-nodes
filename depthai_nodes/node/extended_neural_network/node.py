from typing import Optional, Sequence, Tuple, Union, overload

import depthai as dai
import numpy as np

from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.coordinates_mapper import CoordinatesMapper
from depthai_nodes.node.img_detections_filter import ImgDetectionsFilter
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node.tiles_patcher import TilesPatcher
from depthai_nodes.node.tiling import Tiling

from .config import DetectionFilterConfig, TilingConfig


class ExtendedNeuralNetwork(BaseThreadedHostNode):
    """High-level node that builds a neural-network subgraph with optional tiling,
    coordinate remapping, and detection filtering.

    This node wraps :class:`~depthai_nodes.node.parsing_neural_network.ParsingNeuralNetwork`
    and adds common pipeline conveniences:

    - **Automatic input resizing** to the model input resolution (via ImageManip).
    - **Coordinate remapping** of NN outputs into the input frame's reference
      frame (via :class:`~depthai_nodes.node.coordinates_mapper.CoordinatesMapper`).
    - **Optional detection filtering** by confidence threshold and label allow/deny
      lists (via :class:`~depthai_nodes.node.img_detections_filter.ImgDetectionsFilter`).
      This is only applicable for detection-like outputs (e.g. ``ImgDetections`` /
      ``ImgDetectionsExtended``).
    - **Optional input tiling** where the input frame is split into overlapping tiles,
      inference runs per-tile, and outputs are stitched back together (via
      :class:`~depthai_nodes.node.tiling.Tiling` and
      :class:`~depthai_nodes.node.tiles_patcher.TilesPatcher`).

    Configuration is provided via the fluent methods :meth:`with_tiling` and
    :meth:`with_detections_filter`. The pipeline nodes are constructed only once
    :meth:`build` is called.

    Notes
    -----
    - Only **single-head** models are supported.
    - When tiling is enabled, outputs are combined by :class:`TilesPatcher`
      using the configured IOU threshold.
    - When detection filtering is enabled together with tiling, the filter is
      applied to per-tile outputs before stitching.
    - The final output type depends on the underlying model head / parser.

    Outputs
    -------
    out : dai.Node.Output
        Output stream of the underlying NN pipeline. If filtering is enabled,
        this is the filtered output. If tiling is enabled, this is the stitched
        output from :class:`TilesPatcher`.
    nn_passthrough : dai.Node.Output
        Passthrough frame stream from :class:`ParsingNeuralNetwork`.

    See Also
    --------
    ParsingNeuralNetwork
        Base NN wrapper used for inference and parsing.
    Tiling, TilesPatcher
        Nodes used when tiling is enabled.
    CoordinatesMapper
        Used to remap coordinates into the input frame reference frame.
    ImgDetectionsFilter
        Optional detection filtering node.
    """

    def __init__(self):
        super().__init__()

        self._tiling_config: Optional[TilingConfig] = None
        self._filter_config: Optional[DetectionFilterConfig] = None
        self._input_resize_mode: Optional[dai.ImageManipConfig.ResizeMode] = None
        self._out: Optional[dai.Node.Output] = None

        self.nn: Optional[ParsingNeuralNetwork] = None
        self.tiling: Optional[Tiling] = None
        self.patcher: Optional[TilesPatcher] = None
        self.detections_filter: Optional[ImgDetectionsFilter] = None
        self.nn_resize: Optional[dai.node.ImageManip] = None
        self.img_detections_mapper: Optional[CoordinatesMapper] = None

    def with_tiling(
            self,
            *,
            input_size: tuple[int, int],
            grid_size: tuple[int, int] = (2, 2),
            overlap: float = 0.1,
            iou_threshold: float = 0.2,
            grid_matrix: Optional[np.ndarray] = None,
            global_detection: bool = False,
    ):
        if any([i <= 0 for i in input_size]):
            raise ValueError("Input size must be positive")
        self._tiling_config = TilingConfig(
            input_size=input_size,
            grid_size=grid_size,
            overlap=overlap,
            iou_threshold=iou_threshold,
            grid_matrix=grid_matrix,
            global_detection=global_detection
        )
        return self

    def with_detections_filter(
            self,
            *,
            confidence_threshold: Optional[float] = None,
            labels_to_keep: Optional[Sequence[int]] = None,
            labels_to_reject: Optional[Sequence[int]] = None,
            max_detections: Optional[int] = None,
    ):
        self._filter_config = DetectionFilterConfig(
            confidence_threshold=confidence_threshold,
            labels_to_keep=labels_to_keep,
            labels_to_reject=labels_to_reject,
            max_detections=max_detections,
        )
        return self

    def build(
        self,
        image_input: dai.Node.Output,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
    ) -> "ExtendedNeuralNetwork":
        """Builds the underlying nodes.

        @param image_input: ImgFrame node's input. Frames are automatically resized to fit the neural network input size.
        @type image_input: Node.Input
        @param nn_source: NNModelDescription object containing the HubAI model descriptors, NNArchive object of the model,
            or HubAI model slug in form of <model_slug>:<model_version_slug> or <model_slug>:<model_version_slug>:<model_instance_hash>.
        @type nn_source: Union[dai.NNModelDescription, dai.NNArchive, str]
        @param input_resize_mode: Resize mode used when adapting frames to the model input size.
            This is applied either by ImageManip (non-tiling path) or by the tiling pipeline.
        @type input_resize_mode: dai.ImageManipConfig.ResizeMode
        @rtype: ExtendedNeuralNetwork
        @raise RuntimeError: If the NN model has more than one head.
        @raise ValueError: If NNArchive does not contain input size.
        """
        self._input_resize_mode = input_resize_mode
        if self._tiling_config:
            self._out = self._createTilingPipeline(
                image_input,
                self._tiling_config.input_size,
                self._input_resize_mode,
                nn_source,
            )
        else:
            self._out = self._createBasicPipeline(image_input, self._input_resize_mode, nn_source)
        self._logger.debug("ExtendedNeuralNetwork built")
        return self

    def run(self):
        pass

    def _createBasicPipeline(
        self,
        image_input: dai.Node.Output,
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
    ):
        """Create inner nodes, when tiling is disabled."""

        self._logger.debug("Creating basic pipeline")
        self._logger.debug("Creating ImageManip node for resizing NN input")
        self.nn_resize = self._pipeline.create(dai.node.ImageManip)
        image_input.link(self.nn_resize.inputImage)
        self._logger.debug("Building ParsingNeuralNetwork")
        self.nn = self._pipeline.create(ParsingNeuralNetwork).build(
            self.nn_resize.out, nn_source
        )
        if self.nn._getModelHeadsLen() != 1:
            raise RuntimeError(
                f"ExtendedNeuralNetwork only supports single head models. The model has {self.nn._getModelHeadsLen()} heads."
            )
        nn_w = self.nn._nn_archive.getInputWidth()
        nn_h = self.nn._nn_archive.getInputHeight()
        if nn_w is None or nn_h is None:
            raise ValueError("NNArchive does not contain input size")
        self.nn_resize.initialConfig.setOutputSize(nn_w, nn_h, input_resize_mode)
        self.nn_resize.setMaxOutputFrameSize(nn_w * nn_h * 3)
        self.nn_resize.initialConfig.setFrameType(self._img_frame_type)

        self._logger.debug("Building CoordinatesMapper")
        self.img_detections_mapper = self._pipeline.create(CoordinatesMapper).build(
            to_transformation_input=image_input,
            from_transformation_input=self.nn.out,
        )
        output = self.img_detections_mapper.out
        if self._filter_config:
            self.detections_filter = self._pipeline.create(ImgDetectionsFilter).build(
                self.img_detections_mapper.out,
                labels_to_keep=self._filter_config.labels_to_keep,
                labels_to_reject=self._filter_config.labels_to_reject,
                confidence_threshold=self._filter_config.confidence_threshold,
                max_detections=self._filter_config.max_detections,
            )
            output = self.detections_filter.out
        return output

    def _createTilingPipeline(
        self,
        image_input: dai.Node.Output,
        input_size: Tuple[int, int],
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
    ):
        """Create inner nodes, when tiling is enabled."""

        self._logger.debug("Creating tiling pipeline")
        self.tiling = self._pipeline.create(Tiling)
        self._logger.debug("Building ParsingNeuralNetwork")
        self.nn = self._pipeline.create(ParsingNeuralNetwork).build(
            self.tiling.out, nn_source
        )
        if self.nn._getModelHeadsLen() != 1:
            raise RuntimeError(
                f"ExtendedNeuralNetwork only supports single head models. The model has {self.nn._getModelHeadsLen()} heads."
            )
        nn_size = self.nn._nn_archive.getInputSize()
        if nn_size is None:
            raise ValueError("NNArchive does not contain input size")
        self._logger.debug("Building Tiling")
        self.tiling.build(
            img_output=image_input,
            img_shape=input_size,
            overlap=self._tiling_config.overlap,
            grid_size=self._tiling_config.grid_size,
            resize_mode=input_resize_mode,
            global_detection=self._tiling_config.global_detection,
            grid_matrix=self._tiling_config.grid_matrix,
            nn_shape=nn_size,
        )
        self.tiling.setFrameType(self._img_frame_type)
        patcher_input = self.nn.out
        if self._filter_config:
            self.detections_filter = self._pipeline.create(ImgDetectionsFilter).build(
                self.nn.out,
                labels_to_keep=self._filter_config.labels_to_keep,
                labels_to_reject=self._filter_config.labels_to_reject,
                confidence_threshold=self._filter_config.confidence_threshold,
                max_detections=self._filter_config.max_detections,
            )
            patcher_input = self.detections_filter.out

        self._logger.debug("Building TilesPatcher")
        self.patcher = self._pipeline.create(TilesPatcher).build(
            img_frames=image_input,
            nn=patcher_input,
            conf_thresh=0.0,  # confidence filtering is only done in the filter node if enabled
            iou_thresh=self._tiling_config.iou_threshold,
        )
        return self.patcher.out

    @property
    def out(self):
        if self._out is None:
            raise RuntimeError("ExtendedNeuralNetwork not initialized")
        return self._out

    @property
    def nn_passthrough(self):
        if self.nn is None:
            raise RuntimeError("ExtendedNeuralNetwork not initialized")
        return self.nn.passthrough

    @property
    def config(self) -> dict:
        payload = {
            "Tiling": self._tiling_config.__dict__ if self._tiling_config else None,
            "Detections Filter": self._filter_config.__dict__ if self._filter_config else None,
            "Input Resize Mode": self._input_resize_mode,
        }
        return payload

