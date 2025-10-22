from typing import List, Tuple, Union

import depthai as dai
import numpy as np

from depthai_nodes.message.img_detections import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
)
from depthai_nodes.node.base_host_node import BaseHostNode


class DetectionCropper(BaseHostNode):
    """Handles the cropping of detections from neural network outputs.

    Outputs 1 cropped dai.ImgFrame per detection
    """

    SCRIPT_CONTENT = """
try:
    while True:
        # We receive 1 detection count message and image per frame
        frame = node.inputs['preview'].get()
        det_count_msg = node.inputs['det_count'].get()
        detection_count = det_count_msg.getData()[0]

        # We receive 1 ImageManipConfig message per detection and send it to
        # the cropper ImageManip node along with the frame
        for i in range(detection_count):
            cfg = node.inputs['cfg'].get()
            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)

except Exception as e:
    node.warn(str(e))
"""

    def __init__(self):
        super().__init__()
        self._pipeline = self.getParentPipeline()

        self.cropper_image_manip = self._pipeline.create(dai.node.ImageManip)
        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(self.SCRIPT_CONTENT)

        self._cfg_out = self.createOutput()
        self._det_count_out = self.createOutput()
        self._logger.debug("DetectionCropper initialized")

    def build(
        self,
        detections: dai.Node.Output,
        img_frames: dai.Node.Output,
        output_size: Tuple[int, int],
        resize_mode: dai.ImageManipConfig.ResizeMode,
        padding: float = 0,
    ) -> "DetectionCropper":
        self.link_args(detections)

        self._cfg_out.link(self._script.inputs["cfg"])
        self._det_count_out.link(self._script.inputs["det_count"])
        img_frames.link(self._script.inputs["preview"])

        self._script.outputs["manip_cfg"].link(self.cropper_image_manip.inputConfig)
        self._script.outputs["manip_img"].link(self.cropper_image_manip.inputImage)

        self._output_size = output_size
        self._padding = padding
        self._resize_mode = resize_mode
        self.cropper_image_manip.setMaxOutputFrameSize(
            self._output_size[0] * self._output_size[1] * 3
        )
        self.cropper_image_manip.initialConfig.setOutputSize(
            *self._output_size, self._resize_mode
        )
        self._logger.debug("DetectionCropper built")
        return self

    def process(self, detections: dai.Buffer) -> None:
        assert isinstance(detections, (ImgDetectionsExtended, dai.ImgDetections))
        self._sendDetectionCount(detections)
        for detection in detections.detections:
            if isinstance(detection, ImgDetectionExtended):
                bbox = detection.rotated_rect.getOuterRect()
            else:
                bbox = [detection.xmin, detection.ymin, detection.xmax, detection.ymax]
            cfg = self._generateCropCfg(bbox)
            cfg.setTimestamp(detections.getTimestamp())
            cfg.setTimestampDevice(detections.getTimestampDevice())
            cfg.setSequenceNum(detections.getSequenceNum())
            self._cfg_out.send(cfg)

    def _generateCropCfg(self, bbox: List[float]) -> dai.ImageManipConfig:
        rect = self._getCropRectFromBbox(bbox)
        cfg = dai.ImageManipConfig()
        cfg.addCropRotatedRect(rect, normalizedCoords=True)
        cfg.setOutputSize(self._output_size[0], self._output_size[1], self._resize_mode)
        cfg.setFrameType(self._img_frame_type)
        return cfg

    def _getCropRectFromBbox(self, bbox: List[float]) -> dai.RotatedRect:
        if len(bbox) != 4:
            raise ValueError("Bbox must be a list of 4 values.")

        clamped_bbox = [max(0, min(1, x)) for x in bbox]
        xmin, ymin, xmax, ymax = clamped_bbox
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        rect = dai.RotatedRect()
        rect.center.x = (xmin + xmax) / 2
        rect.center.y = (ymin + ymax) / 2
        rect.size.width = xmax - xmin
        rect.size.height = ymax - ymin
        rect.size.width = rect.size.width + self._padding * 2
        rect.size.height = rect.size.height + self._padding * 2
        rect.angle = 0

        return rect

    def _sendDetectionCount(
        self, detections: Union[ImgDetectionsExtended, dai.ImgDetections]
    ) -> None:
        buff = dai.Buffer()
        buff.setData(np.array([len(detections.detections)], dtype=np.uint8))
        self._det_count_out.send(buff)

    @property
    def out(self) -> dai.Node.Output:
        return self.cropper_image_manip.out
