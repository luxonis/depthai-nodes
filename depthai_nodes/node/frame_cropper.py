from string import Template
from typing import Optional, Tuple

import depthai as dai

from depthai_nodes.message.collection import Collection
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode


class FrameCropper(BaseThreadedHostNode):
    """Handles the cropping of detections from neural network outputs.

    Outputs 1 cropped dai.ImgFrame per detection
    """

    SCRIPT_CONTENT = Template(
        """
        try:
            OUT_WIDTH = $OUT_WIDTH
            OUT_HEIGHT = $OUT_HEIGHT
            PADDING = $PADDING
            FRAME_TYPE = dai.ImgFrame.Type.$FRAME_TYPE
            while True:
                # We receive 1 detection count message and image per frame
                frame = node.inputs['inputImage'].get()
                img_detections = node.inputs['inputImgDetections'].get()
                for det in range(img_detections.detections):
                    cfg = dai.ImageManipConfig()
                    cfg.addCropRotatedRect(rect=det.getBoundingBox(), normalizedCoords=True)
                    cfg.setTimestamp(det.getTimestamp())
                    cfg.setTimestampDevice(det.getTimestampDevice())
                    cfg.setSequenceNum(det.getSequenceNum())
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
