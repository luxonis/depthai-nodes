from datetime import timedelta
from string import Template
from typing import cast

import depthai as dai

from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode


class FrameCropper(BaseThreadedHostNode):
    """A host node that crops detection regions from frames and outputs one cropped
    ``dai.ImgFrame`` per region.

    ``FrameCropper`` is a convenience wrapper around an internal ``dai.node.ImageManip``
    configured for cropping + resizing. It supports
    two input modes:

    - **fromImgDetections**: Provide ``dai.ImgDetections``
      and the node will generate ``dai.ImageManipConfig`` messages for each detection
      via a ``dai.node.Script`` node. Each config is paired with the corresponding input
      frame, producing one cropped output frame per detection.
    - **fromManipConfigs**: Provide an upstream stream of cropping configs packed
      in ``dai.MessageGroup`` messages. An on-device ``dai.node.Script`` node pairs each
      config with the current frame and forwards them to the internal
      ``dai.node.ImageManip``.

    Configuration is provided via ``fromImgDetections`` or ``fromManipConfigs``. The
    pipeline nodes are constructed only once ``build`` is called.

    Note:
        - Exactly one configuration path must be selected: only one of
          ``fromImgDetections`` and ``fromManipConfigs`` can be used.
        - Output frames are always resized to ``outputSize`` using the provided
          ``resizeMode`` (default: ``CENTER_CROP``).
        - In ``fromImgDetections`` mode, a ``dai.node.Script`` node drives the
          cropping by emitting one ``dai.ImageManipConfig`` per detection.
        - In ``fromManipConfigs`` mode, the ``inputManipConfigs`` stream **must**
          output ``dai.MessageGroup`` messages where each value is a
          ``dai.ImageManipConfig``. Key naming is arbitrary.

    Use ``fromImgDetections(padding=0.0)`` to set optional padding around each
    detection region. ``build(outputSize, resizeMode)`` sets the crop size and
    resize mode.

    Outputs:

    * ``out : dai.Node.Output``: Stream of cropped ``dai.ImgFrame`` messages. One output
      frame is
      produced per crop configuration (per detection in ``fromImgDetections``
      mode; per config in the received ``MessageGroup`` in ``fromManipConfigs`` mode).

    See also:

    dai.node.ImageManip
        Node used to perform cropping and resizing.
    dai.ImageManipConfig
        Cropping configuration messages forwarded to ImageManip.
    dai.ImgDetections
        Detection message type used in ``fromImgDetections`` mode.
    dai.MessageGroup
        Message type expected by ``fromManipConfigs``.
    """

    IMG_DETECTIONS_SCRIPT_CONTENT = Template(
        """
        def pad_rotated_rect(rot: RotatedRect, p: float) -> RotatedRect:
            return RotatedRect(
                rot.center,
                Size2f(width=rot.size.width + 2*p, height=rot.size.height + 2*p, normalized=True),
                rot.angle
            )
        try:
            OUT_WIDTH = $OUT_WIDTH
            OUT_HEIGHT = $OUT_HEIGHT
            PADDING = $PADDING
            FRAME_TYPE = ImgFrame.Type.$FRAME_TYPE
            RESIZE_MODE = ImageManipConfig.ResizeMode.$RESIZE_MODE
            while True:
                # Sync guarantees the image and detections belong to the same source frame.
                group = node.inputs['synced'].get()
                frame = group['frame']
                img_detections = group['detections']
                for det in img_detections.detections:
                    rot_rect = det.getBoundingBox()
                    cfg = ImageManipConfig()
                    cfg.addCropRotatedRect(rect=pad_rotated_rect(rot_rect, PADDING), normalizedCoords=True)
                    cfg.setTimestamp(img_detections.getTimestamp())
                    cfg.setTimestampDevice(img_detections.getTimestampDevice())
                    cfg.setSequenceNum(img_detections.getSequenceNum())
                    cfg.setOutputSize(OUT_WIDTH, OUT_HEIGHT, RESIZE_MODE)
                    cfg.setFrameType(FRAME_TYPE)
                    node.outputs['manip_img'].send(frame)
                    node.outputs['manip_cfg'].send(cfg)

        except Exception as e:
            node.error(str(e))
        """
    )

    MANIP_CONFIGS_SCRIPT_CONTENT = Template(
        """
        try:
            latest_cfgs = node.inputs['inputManipConfigs'].get()
            while True:
                frame = node.inputs['inputImage'].get()
                cfgs = node.inputs['inputManipConfigs'].tryGet()
                if cfgs:
                    latest_cfgs = cfgs
                for key, cfg in latest_cfgs:
                    node.outputs['manip_cfg'].send(cfg)
                    node.outputs['manip_img'].send(frame)
        except Exception as e:
            node.error(str(e))
        """
    )

    SYNCED_MANIP_CONFIGS_SCRIPT_CONTENT = Template(
        """
        try:
            while True:
                # Sync guarantees the image and config group belong to the same source frame.
                group = node.inputs['synced'].get()
                frame = group['frame']
                cfgs = group['configs']
                for key, cfg in cfgs:
                    node.outputs['manip_cfg'].send(cfg)
                    node.outputs['manip_img'].send(frame)
        except Exception as e:
            node.error(str(e))
        """
    )

    def __init__(self):
        super().__init__()
        self._cropper_image_manip = self.createSubnode(dai.node.ImageManip)
        self._version_selected = False
        self._output_size: tuple[int, int] | None = None  # width, height
        self._sync: dai.node.Sync | None = None
        self._sync_threshold: timedelta = timedelta(milliseconds=10)

        # when fromImgDetections is used script node can work on ImgDetections directly
        self._script: dai.node.Script | None = None
        self._input_img_detections: dai.Node.Output | None = None
        self._padding: float | None = None
        self._resize_mode: dai.ImageManipConfig.ResizeMode | None = None

        # when fromManipConfigs is used, script node receives MessageGroup of precomputed configs
        self._input_manip_configs: dai.Node.Output | None = None
        self._wait_for_cfg = False
        self._logger.debug("FrameCropper initialized")

    @property
    def out(self):
        """Return the cropped frame output stream."""
        return self._cropper_image_manip.out

    def fromImgDetections(
        self,
        inputImgDetections: dai.Node.Output,
        outputSize: tuple[int, int],
        resizeMode: dai.ImageManipConfig.ResizeMode = dai.ImageManipConfig.ResizeMode.CENTER_CROP,
        padding: float = 0.0,
        syncThreshold: timedelta = timedelta(milliseconds=10),
    ) -> "FrameCropper":
        """Select detection-driven cropping before calling ``build()``.

        Args:
            inputImgDetections: Output stream of ``dai.ImgDetections`` to synchronize
                with image frames.
            outputSize: Crop output size as ``(width, height)`` pixels.
            resizeMode: ImageManip resize policy applied to each crop.
            padding: Normalized padding added to each side of the detection region.
            syncThreshold: Maximum timestamp difference used to synchronize detections
                and frames.

        Returns:
            This node for fluent configuration.

        Raises:
            RuntimeError: If either crop configuration mode was already selected.

        Note:
            Produces one crop per detection. The image stream is connected later by
            ``build()``.
        """
        if self._version_selected:
            raise RuntimeError(
                "FrameCropper was already configured using the `fromManipConfigs` method. "
                "Only one of `fromImgDetections` and `fromManipConfigs` can be used."
            )
        self._version_selected = True
        self._input_img_detections = inputImgDetections
        self._output_size = outputSize
        self._padding = padding
        self._resize_mode = resizeMode
        self._sync_threshold = syncThreshold
        self._cropper_image_manip.setMaxOutputFrameSize(
            self._output_size[0] * self._output_size[1] * 3
        )
        self._cropper_image_manip.initialConfig.setOutputSize(
            *self._output_size, mode=resizeMode
        )
        return self

    def fromManipConfigs(
        self,
        inputManipConfigs: dai.Node.Output,
        maxOutputFrameSize: int,
        waitForConfig: bool,
        syncThreshold: timedelta | None = None,
    ) -> "FrameCropper":
        """Select cropping from groups of ImageManip configuration messages.

        Args:
            inputManipConfigs: Stream of ``dai.MessageGroup`` objects whose values are
                ``dai.ImageManipConfig`` messages. Group keys are arbitrary.
            maxOutputFrameSize: Maximum output image buffer size in bytes.
            waitForConfig: If true, synchronize each frame with a configuration group.
                Otherwise, reuse the latest group for subsequent frames.
            syncThreshold: Optional timestamp tolerance. May only be set when
                ``waitForConfig`` is true.

        Returns:
            This node for fluent configuration.

        Raises:
            RuntimeError: If a configuration mode was already selected, or a sync
                threshold is supplied without waiting for configuration.
        """
        if self._version_selected:
            raise RuntimeError(
                "FrameCropper was already configured using the `fromManipConfig` method. "
                "Only one of `fromImgDetections` and `fromManipConfigs` can be used."
            )
        if syncThreshold is not None and waitForConfig is False:
            raise RuntimeError(
                "syncThreshold can only be used when waitForConfig is True."
            )
        elif syncThreshold is not None:
            self._sync_threshold = syncThreshold
        self._version_selected = True
        self._input_manip_configs = inputManipConfigs
        self._wait_for_cfg = waitForConfig
        self._cropper_image_manip.setMaxOutputFrameSize(maxOutputFrameSize)
        return self

    def build(
        self,
        inputImage: dai.Node.Output,
    ) -> "FrameCropper":
        """Connect image input and construct the configured crop pipeline.

        Args:
            inputImage: Image stream to crop. Call ``fromImgDetections()`` or
                ``fromManipConfigs()`` first.

        Returns:
            This node, with cropped frames available on ``out``.

        Raises:
            RuntimeError: If no crop configuration mode has been selected.
        """
        if not self._version_selected:
            raise RuntimeError(
                "Configure the FrameCropper by calling one of the `fromImgDetections` or `fromManipConfigs` methods first."
            )

        self._cropper_image_manip.inputConfig.setWaitForMessage(waitForMessage=True)
        if self._input_img_detections is not None:
            self._build_detections_cropper(input_image=inputImage)
        else:
            self._build_manip_configs_cropper(input_image=inputImage)
        self._logger.debug("FrameCropper built")
        return self

    def run(self) -> None:
        """No-op because cropping is driven entirely by on-device Script nodes."""
        return  # Both fromImgDetections and fromManipConfigs use on-device Script

    def _build_detections_cropper(self, input_image: dai.Node.Output):
        assert_msg = "Configure the FrameCropper by calling one of the `fromImgDetections` or `fromManipConfigs` methods first."
        assert self._input_img_detections is not None, assert_msg
        assert self._output_size is not None, assert_msg
        assert self._padding is not None, assert_msg
        assert self._resize_mode is not None, assert_msg

        self._script = cast(dai.node.Script, self.createSubnode(dai.node.Script))
        self._script.setScript(
            self.IMG_DETECTIONS_SCRIPT_CONTENT.substitute(
                {
                    "OUT_WIDTH": self._output_size[0],
                    "OUT_HEIGHT": self._output_size[1],
                    "PADDING": self._padding,
                    "FRAME_TYPE": self._img_frame_type.name,
                    "RESIZE_MODE": self._resize_mode.name,
                }
            )
        )
        self._sync = cast(dai.node.Sync, self.createSubnode(dai.node.Sync))
        self._sync.setSyncThreshold(self._sync_threshold)
        self._sync.setSyncAttempts(-1)
        input_image.link(self._sync.inputs["frame"])
        self._input_img_detections.link(self._sync.inputs["detections"])
        self._sync.out.link(self._script.inputs["synced"])
        self._script.outputs["manip_cfg"].link(self._cropper_image_manip.inputConfig)
        self._script.outputs["manip_img"].link(self._cropper_image_manip.inputImage)

    def _build_manip_configs_cropper(self, input_image: dai.Node.Output):
        assert self._input_manip_configs is not None

        self._script = cast(dai.node.Script, self.createSubnode(dai.node.Script))
        if self._wait_for_cfg:
            script_content = self.SYNCED_MANIP_CONFIGS_SCRIPT_CONTENT.substitute()
            self._sync = cast(dai.node.Sync, self.createSubnode(dai.node.Sync))
            self._sync.setSyncThreshold(self._sync_threshold)
            self._sync.setSyncAttempts(-1)
            input_image.link(self._sync.inputs["frame"])
            self._input_manip_configs.link(self._sync.inputs["configs"])
            self._sync.out.link(self._script.inputs["synced"])
        else:
            script_content = self.MANIP_CONFIGS_SCRIPT_CONTENT.substitute()
            input_image.link(self._script.inputs["inputImage"])
            self._input_manip_configs.link(self._script.inputs["inputManipConfigs"])
        self._script.setScript(script_content)
        self._script.outputs["manip_cfg"].link(self._cropper_image_manip.inputConfig)
        self._script.outputs["manip_img"].link(self._cropper_image_manip.inputImage)
