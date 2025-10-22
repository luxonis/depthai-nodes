import argparse

import depthai as dai
import numpy as np

from depthai_nodes.logging import get_logger
from depthai_nodes.node.detection_cropper import DetectionCropper

parser = argparse.ArgumentParser()
parser.add_argument("-ip", type=str, default="", help="IP of the device")
args = parser.parse_args()

logger = get_logger(__name__)

try:
    device = dai.Device(dai.DeviceInfo(args.ip))
    logger.debug(f"(1) Connected to device with IP/mxid: {args.ip}")
except Exception as e:
    logger.warning(e)
    logger.warning("Can't connect to the device with IP/mxid: %s", args.ip)
    exit(6)


class DataProvider(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.image_frame_out = self.createOutput()
        self.detection_out = self.createOutput()

    def build(self, cam_frame: dai.Node.Output) -> "DataProvider":
        self.link_args(cam_frame)
        return self

    def process(self, cam_frame: dai.ImgFrame) -> None:
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        image[160:480, 160:480] = [255, 255, 255]

        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(image, dai.ImgFrame.Type.BGR888i)
        img_frame.setWidth(640)
        img_frame.setHeight(640)
        img_frame.setTimestamp(cam_frame.getTimestamp())
        img_frame.setTransformation(cam_frame.getTransformation())
        img_frame.setSequenceNum(cam_frame.getSequenceNum())

        nn_out = dai.ImgDetections()
        img_det = dai.ImgDetection()
        img_det.confidence = 0.5
        img_det.label = 0
        img_det.xmin = 0.25
        img_det.ymin = 0.25
        img_det.xmax = 0.75
        img_det.ymax = 0.75
        nn_out.detections = [img_det]

        nn_out.setTimestamp(cam_frame.getTimestamp())
        nn_out.setTransformation(cam_frame.getTransformation())
        nn_out.setSequenceNum(cam_frame.getSequenceNum())

        self.image_frame_out.send(img_frame)
        self.detection_out.send(nn_out)


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    platform = device.getPlatform()

    logger.debug(f"(2) Platform: {platform}")

    img_frame_type = (
        dai.ImgFrame.Type.BGR888p
        if platform == dai.Platform.RVC2
        else dai.ImgFrame.Type.BGR888i
    )
    logger.debug(f"(3) Image frame type: {img_frame_type}")

    logger.debug("(4) Creating camera node.")
    cam = pipeline.create(dai.node.Camera).build()
    logger.debug("(4) Camera node created.")
    cam_out = cam.requestOutput((640, 640), type=img_frame_type)
    logger.debug("(5) Camera output created.")

    data_provider = pipeline.create(DataProvider).build(cam_out)
    logger.debug("(6) Data provider created.")

    detection_cropper = pipeline.create(DetectionCropper).build(
        data_provider.detection_out,
        data_provider.image_frame_out,
        (320, 320),
        padding=0,
        resize_mode=dai.ImageManipConfig.ResizeMode.STRETCH,
    )
    logger.debug("(7) Detection cropper created.")

    detection_cropper_out = detection_cropper.out.createOutputQueue()
    logger.debug("(8) Detection cropper output created.")

    logger.debug("(9) Pipeline created.")

    pipeline.start()
    logger.debug("(10) Pipeline started.")

    while pipeline.isRunning():
        detection_cropper_out: dai.ImgFrame = detection_cropper_out.get()
        logger.debug("(11) Detection cropper output received.")
        pipeline.stop()
        logger.debug("(12) Pipeline stopped.")
        break
