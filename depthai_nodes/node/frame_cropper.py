from string import Template
from typing import Optional, Tuple

import depthai as dai

from depthai_nodes.message.collection import Collection
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode


class FrameCropper(BaseThreadedHostNode):
    """A host node that crops detection regions from frames and outputs one
    cropped :class:`dai.ImgFrame` per region.

    `FrameCropper` is a convenience wrapper around an internal
    :class:`dai.node.ImageManip` configured for cropping + resizing. It supports
    two input modes:

    - **fromImgDetections**: Provide :class:`dai.ImgDetections`
      and the node will generate :class:`dai.ImageManipConfig` messages for each
      detection via a :class:`dai.node.Script` node. Each config is paired with
      the corresponding input frame, producing one cropped output frame per
      detection.
    - **fromManipConfigs**: Provide an upstream stream of cropping configs packed
      in :class:`~depthai_nodes.message.collection.Collection` messages. In this
      mode `FrameCropper` runs host-side and forwards each
      :class:`dai.ImageManipConfig` together with the current frame to the
      internal :class:`dai.node.ImageManip`.

    Configuration is provided via :meth:`fromImgDetections` or
    :meth:`fromManipConfigs`. The pipeline nodes are constructed only once
    :meth:`build` is called.

    Notes
    -----
    - Exactly one configuration path must be selected: only one of
      :meth:`fromImgDetections` and :meth:`fromManipConfigs` can be used.
    - Output frames are always resized to `outputSize` using the provided
      `resizeMode` (default: ``CENTER_CROP``).
    - In `fromImgDetections` mode, a :class:`dai.node.Script` node drives the
      cropping by emitting one :class:`dai.ImageManipConfig` per detection.
    - In `fromManipConfigs` mode, the `inputManipConfigs` stream **must**
      output :class:`~depthai_nodes.message.collection.Collection` messages
      containing only :class:`dai.ImageManipConfig` items.

    Parameters
    ----------
    fromImgDetections(padding=0.0)
        Optional padding factor applied around each detection region.
    build(outputSize, resizeMode)
        Sets the crop output size and the resize mode used by ImageManip.

    Outputs
    -------
    out : dai.Node.Output
        Stream of cropped :class:`dai.ImgFrame` messages. One output frame is
        produced per crop configuration (per detection in `fromImgDetections`
        mode; per item in the received `Collection` in `fromManipConfigs` mode).

    See Also
    --------
    dai.node.ImageManip
        Node used to perform cropping and resizing.
    dai.ImageManipConfig
        Cropping configuration messages forwarded to ImageManip.
    dai.ImgDetections
        Detection message type used in `fromImgDetections` mode.
    Collection
        Container type expected by `fromManipConfigs`.
    """

    SCRIPT_CONTENT = Template(
        """
        try:
            OUT_WIDTH = $OUT_WIDTH
            OUT_HEIGHT = $OUT_HEIGHT
            PADDING = $PADDING
            FRAME_TYPE = ImgFrame.Type.$FRAME_TYPE
            while True:
                # We receive 1 detection count message and image per frame
                frame = node.inputs['inputImage'].get()
                img_detections = node.inputs['inputImgDetections'].get()
                for det in img_detections.detections:
                    cfg = ImageManipConfig()
                    cfg.addCropRotatedRect(rect=det.getBoundingBox(), normalizedCoords=True)
                    cfg.setTimestamp(img_detections.getTimestamp())
                    cfg.setTimestampDevice(img_detections.getTimestampDevice())
                    cfg.setSequenceNum(img_detections.getSequenceNum())
                    cfg.setOutputSize(OUT_WIDTH * OUT_HEIGHT * 3)
                    node.outputs['manip_cfg'].send(cfg)
                    node.outputs['manip_img'].send(frame)
    
        except Exception as e:
            node.error(str(e))
        """
    )

    def __init__(self):
        super().__init__()
        self._cropper_image_manip = self._pipeline.create(dai.node.ImageManip)
        self._version_selected = False
        self._output_size: Optional[Tuple[int, int]] = None  # width, height

        # when fromImgDetections is used script node can work on ImgDetections directly
        self._script: Optional[dai.node.Script] = None
        self._input_img_detections: Optional[dai.Node.Output] = None
        self._padding: Optional[float] = None

        # when fromManipConfigs is used, need to manually process the Collection containing ImageManipConfigs
        self._image_input: Optional[dai.Node.Input] = None
        self._input_manip_configs: Optional[dai.Node.Output] = None
        self._manip_configs_input: Optional[dai.Node.Input] = None
        self._image_output: Optional[dai.Node.Output] = None
        self._config_output: Optional[dai.Node.Output] = None
        self._logger.debug("FrameCropper initialized")

    @property
    def out(self):
        return self._cropper_image_manip.out

    def fromImgDetections(self, inputImgDetections: dai.Node.Output, padding: float = 0.) -> "FrameCropper":
        """Configure cropping from an ImgDetections stream.

        In this mode the node generates ImageManipConfig messages per detection (via Script)
        and outputs one cropped ImgFrame per detection. `padding` expands the crop region.
        """
        if self._version_selected is True:
            raise RuntimeError(
                f"FrameCropper was already configured using the `fromManipConfigs` method. "
                f"Only one of `fromImgDetections` and `fromManipConfigs` can be used."
            )
        self._version_selected = True
        self._input_img_detections = inputImgDetections
        self._padding = padding
        return self

    def fromManipConfigs(self, inputManipConfigs: dai.Node.Output) -> "FrameCropper":
        """Configure cropping from a stream of precomputed ImageManipConfig collections.

        Expects `inputManipConfigs` to output Collection[ImageManipConfig]. The node will
        forward each config paired with the current frame to ImageManip.
        """
        if self._version_selected is True:
            raise RuntimeError(
                f"FrameCropper was already configured using the `fromManipConfig` method. "
                f"Only one of `fromImgDetections` and `fromManipConfigs` can be used."
            )
        self._version_selected = True
        self._input_manip_configs = inputManipConfigs
        return self

    def build(
        self,
        inputImage: dai.Node.Output,
        outputSize: Tuple[int, int],
        resizeMode: dai.ImageManipConfig.ResizeMode = dai.ImageManipConfig.ResizeMode.CENTER_CROP,
    ) -> "FrameCropper":
        """Build the internal pipeline and set output size / resize behavior.

        Requires that exactly one configuration path was selected via `fromImgDetections`
        or `fromManipConfigs` before calling. Returns `self` for fluent chaining.
        """
        if self._version_selected is False:
            raise RuntimeError(
                f"Configure the FrameCropper by calling one of the `fromImgDetections` or `fromManipConfigs` methods first."
            )
        self._output_size = outputSize
        self._cropper_image_manip.setMaxOutputFrameSize(self._output_size[0] * self._output_size[1] * 3)
        self._cropper_image_manip.initialConfig.setOutputSize(*self._output_size, mode=resizeMode)
        self._cropper_image_manip.inputConfig.setWaitForMessage(waitForMessage=True)
        if self._input_img_detections is not None:
            self._build_detections_cropper(input_image=inputImage)
        else:
            self._build_frame_cropper()
        self._logger.debug("FrameCropper built")
        return self

    def _build_detections_cropper(self, input_image: dai.Node.Output):
        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(
            self.SCRIPT_CONTENT.substitute(
                {
                    "OUT_WIDTH": self._output_size[0],
                    "OUT_HEIGHT": self._output_size[1],
                    "PADDING": self._padding,
                    "FRAME_TYPE": self._img_frame_type.name,
                }
            )
        )
        input_image.link(self._script.inputs["inputImage"])
        self._input_img_detections.link(self._script.inputs["inputImgDetections"])
        self._script.outputs["manip_cfg"].link(self._cropper_image_manip.inputConfig)
        self._script.outputs["manip_img"].link(self._cropper_image_manip.inputImage)

    def _build_frame_cropper(self, input_image: dai.Node.Output):
        self._image_input = self.createInput(blocking=True)
        self._manip_configs_input = self.createInput(blocking=True)
        self._image_output = self.createOutput()
        self._config_output = self.createOutput()

        input_image.link(self._image_input)
        self._input_manip_configs.link(self._manip_configs_input)
        self._image_output.link(self._cropper_image_manip.inputImage)
        self._config_output.link(self._cropper_image_manip.inputConfig)

    def run(self) -> None:
        if self._input_img_detections is not None:
            return
        image: dai.ImgFrame = self._image_input.get()  # noqa
        manip_configs: Collection[dai.ImageManipConfig] = self._manip_configs_input.get()  # noqa
        assert isinstance(manip_configs, Collection), \
            (f"When FrameCropper is configured using `fromManipConfigs` the `inputManipConfigs` must output messages of type `Collection`."
             f" Received: {type(manip_configs)}")

        for manip_config in manip_configs.items:
            assert isinstance(manip_config, dai.ImageManipConfig), \
                (f"FrameCropper configured using `fromManipConfigs` needs the `Collection` message to contain only `ImageManipConfig`s. "
                 f"Received {[type(m) for m in manip_configs.items]}")
            self._config_output.send(manip_config)
            self._image_output.send(image)
