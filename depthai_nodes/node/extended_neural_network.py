from typing import List, Literal, Optional, Tuple, Union, overload

import depthai as dai
import numpy as np

from depthai_nodes.logging import get_logger
from depthai_nodes.node import (
    ImgDetectionsFilter,
    ParsingNeuralNetwork,
    TilesPatcher,
    Tiling,
)
from depthai_nodes.node.img_detections_mapper import ImgDetectionsMapper


class ExtendedNeuralNetwork(dai.node.ThreadedHostNode):
    """Node that wraps the ParsingNeuralNetwork node and adds following capabilities:
    1. Automatic input resizing to the neural network input size.
    2. Remapping of detection coordinates from neural network output to input frame coordinates.
    3. Neural network output filtering based on confidence threshold and labels.
    (Only supported for ImgDetectionsExtended and ImgDetections messages).
    4. Input tiling.

    Supports only single model heads.

    Attributes
    ----------
    out : Node.Output
        Neural network output. Detections are remapped to the input frame coordinates.
    nn_passthrough : Node.Output
        Neural network frame passthrough.
    """

    IMG_FRAME_TYPES = {
        dai.Platform.RVC2: dai.ImgFrame.Type.BGR888p,
        dai.Platform.RVC4: dai.ImgFrame.Type.BGR888i,
    }

    def __init__(self):
        super().__init__()

        self._platform = self.getParentPipeline().getDefaultDevice().getPlatform()
        try:
            self._img_frame_type = self.IMG_FRAME_TYPES[self._platform]
        except KeyError as e:
            raise ValueError(
                f"No dai.ImgFrame.Type defined for platform {self._platform}."
            ) from e

        self._logger = get_logger(self.__class__.__name__)

        self._confidence_threshold = None
        self._labels_to_keep = None
        self._labels_to_reject = None
        self._max_detections = None
        self._tiling_grid_size = (2, 2)
        self._tiling_overlap = 0.1
        self._tiling_global_detection = False
        self._tiling_grid_matrix = None
        self._tiling_iou_threshold = 0.2

        self._pipeline = self.getParentPipeline()
        self.nn: Optional[ParsingNeuralNetwork] = None
        self.tiling: Optional[Tiling] = None
        self.patcher: Optional[TilesPatcher] = None
        self.detections_filter: Optional[ImgDetectionsFilter] = None
        self.nn_resize: Optional[dai.node.ImageManip] = None
        self.img_detections_mapper: Optional[ImgDetectionsMapper] = None
        self._out: Optional[dai.Node.Output] = None

    @overload
    def build(
        self,
        input: dai.Node.Output,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
        enable_tiling: Literal[False] = False,
        input_size: None = None,
        enable_detection_filtering: bool = False,
    ) -> "ExtendedNeuralNetwork":
        ...

    @overload
    def build(
        self,
        input: dai.Node.Output,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
        enable_tiling: Literal[True],
        input_size: Tuple[int, int],
        enable_detection_filtering: bool = False,
    ) -> "ExtendedNeuralNetwork":
        ...

    def build(
        self,
        input: dai.Node.Output,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
        enable_tiling: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        enable_detection_filtering: bool = False,
    ) -> "ExtendedNeuralNetwork":
        """Builds the underlying nodes.

        @param input: ImgFrame node's input. Frames are automatically resized to fit the neural network input size.
        @type input: Node.Input
        @param nn_source: NNModelDescription object containing the HubAI model descriptors, NNArchive object of the model, or HubAI model slug in form of <model_slug>:<model_version_slug> or <model_slug>:<model_version_slug>:<model_instance_hash>.
        @type nn_source: Union[dai.NNModelDescription, dai.NNArchive, str]
        @param input_resize_mode: Resize mode for the neural network input.
        @type input_resize_mode: dai.ImageManipConfig.ResizeMode
        @param enable_tiling: If True, enables tiling.
        @type enable_tiling: bool
        @param input_size: ImgFrame input size for tiling. Must be provided if tiling is enabled.
        @type input_size: Tuple[int, int]
        @param enable_detection_filtering: If True, enables detection filtering based on labels and confidence threshold
            (only supported for ImgDetectionsExtended and ImgDetections messages).
        @type enable_detection_filtering: bool
        @return: Returns the ExtendedNeuralNetwork object.
        @rtype: ExtendedNeuralNetwork
        @raise ValueError: If tiling is enabled and input_size is not provided.
        @raise ValueError: If NNArchive does not contain input size.
        """
        if enable_tiling:
            if input_size is None:
                raise ValueError("Input size must be provided for tiling")
            nn_out = self._createTilingPipeline(
                input,
                input_size,
                input_resize_mode,
                nn_source,
            )
        else:
            nn_out = self._createBasicPipeline(input, input_resize_mode, nn_source)
        if enable_detection_filtering:
            self.detections_filter = self._pipeline.create(ImgDetectionsFilter).build(
                nn_out,
                labels_to_keep=self._labels_to_keep,  # type: ignore
                labels_to_reject=self._labels_to_reject,  # type: ignore
                confidence_threshold=self._confidence_threshold,
                max_detections=self._max_detections,  # type: ignore
            )
            self._out = self.detections_filter.out
        else:
            self.detections_filter = None
            self._out = nn_out
        return self

    def run(self):
        pass

    def _createBasicPipeline(
        self,
        input: dai.Node.Output,
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
    ):
        """Create inner nodes, when tiling is disabled."""

        self.nn_resize = self._pipeline.create(dai.node.ImageManip)
        input.link(self.nn_resize.inputImage)
        self.nn = self._pipeline.create(ParsingNeuralNetwork).build(
            self.nn_resize.out, nn_source
        )
        nn_w = self.nn._nn_archive.getInputWidth()
        nn_h = self.nn._nn_archive.getInputHeight()
        if nn_w is None or nn_h is None:
            raise ValueError("NNArchive does not contain input size")
        self.nn_resize.initialConfig.setOutputSize(nn_w, nn_h, input_resize_mode)
        self.nn_resize.setMaxOutputFrameSize(nn_w * nn_h * 3)
        self.nn_resize.initialConfig.setFrameType(self._img_frame_type)

        self.img_detections_mapper = self._pipeline.create(ImgDetectionsMapper).build(
            input, self.nn.out
        )
        return self.img_detections_mapper.out

    def _createTilingPipeline(
        self,
        input: dai.Node.Output,
        input_size: Tuple[int, int],
        input_resize_mode: dai.ImageManipConfig.ResizeMode,
        nn_source: Union[dai.NNModelDescription, dai.NNArchive, str],
    ):
        """Create inner nodes, when tiling is enabled."""

        self.tiling = self._pipeline.create(Tiling)
        self.nn = self._pipeline.create(ParsingNeuralNetwork).build(
            self.tiling.out, nn_source
        )
        nn_size = self.nn._nn_archive.getInputSize()
        if nn_size is None:
            raise ValueError("NNArchive does not contain input size")
        self.tiling.build(
            img_output=input,
            img_shape=input_size,
            overlap=self._tiling_overlap,
            grid_size=self._tiling_grid_size,
            resize_mode=input_resize_mode,
            global_detection=self._tiling_global_detection,
            grid_matrix=self._tiling_grid_matrix,
            nn_shape=nn_size,
        )
        self.tiling.setFrameType(self._img_frame_type)
        self.patcher = self._pipeline.create(TilesPatcher).build(
            img_frames=input,
            nn=self.nn.out,
            conf_thresh=self._confidence_threshold or 0.0,
            iou_thresh=self._tiling_iou_threshold,
        )
        return self.patcher.out

    def setTilingGridSize(self, grid_size: Tuple[int, int]) -> None:
        """Set grid size for tiling.

        Only used if tiling is enabled.
        """

        self._tiling_grid_size = grid_size
        if self.tiling is not None:
            self.tiling.setGridSize(grid_size)

    def setTilingOverlap(self, overlap: float) -> None:
        """Set tile overlap.

        Only used if tiling is enabled.
        """

        self._tiling_overlap = overlap
        if self.tiling is not None:
            self.tiling.setOverlap(overlap)

    def setTilingGlobalDetection(self, global_detection: bool) -> None:
        """Set global detection flag for tiling.

        Only used if tiling is enabled.
        """

        self._tiling_global_detection = global_detection
        if self.tiling is not None:
            self.tiling.setGlobalDetection(global_detection)

    def setTilingGridMatrix(self, grid_matrix: Union[np.ndarray, List, None]) -> None:
        """Set grid matrix for tiling.

        Only used if tiling is enabled.
        """

        self._tiling_grid_matrix = grid_matrix
        if self.tiling is not None:
            self.tiling.setGridMatrix(grid_matrix)

    def setLabels(self, labels: List[int] | None, keep: bool) -> None:
        """Set labels to keep or reject."""

        if keep:
            self._labels_to_keep = labels
        else:
            self._labels_to_reject = labels
        if self.detections_filter is not None:
            self.detections_filter.setLabels(labels, keep)  # type: ignore

    def setMaxDetections(self, max_detections: int) -> None:
        """Set maximum number of detections to keep."""

        self._max_detections = max_detections
        if self.detections_filter is not None:
            self.detections_filter.setMaxDetections(max_detections)

    def setConfidenceThreshold(self, confidence_threshold: float) -> None:
        """Set confidence threshold."""

        self._confidence_threshold = confidence_threshold
        if self.detections_filter is not None:
            self.detections_filter.setConfidenceThreshold(confidence_threshold)
        if self.patcher is not None:
            self.patcher.setConfidenceThreshold(confidence_threshold)

    @property
    def out(self):
        if self._out is None:
            raise RuntimeError("Stage1Node not initialized")
        return self._out

    @property
    def nn_passthrough(self):
        if self.nn is None:
            raise RuntimeError("Stage1Node not initialized")
        return self.nn.passthrough
