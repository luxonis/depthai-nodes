import argparse

import depthai as dai

from depthai_nodes.logging import get_logger
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node.tiles_patcher import TilesPatcher
from depthai_nodes.node.tiling import Tiling

parser = argparse.ArgumentParser()
parser.add_argument("-ip", type=str, default="", help="IP of the device")
args = parser.parse_args()

IMG_SHAPE = (1280, 720)
FPS_LIMIT = 10

logger = get_logger(__name__)

try:
    device = dai.Device(dai.DeviceInfo(args.ip))
    logger.debug(f"(1) Connected to device with IP/mxid: {args.ip}")
except Exception as e:
    logger.warning(e)
    logger.warning("Can't connect to the device with IP/mxid: %s", args.ip)
    exit(6)


with dai.Pipeline(device) as pipeline:
    logger.debug("(2) Creating pipeline...")

    platform = device.getPlatform()

    logger.debug(f"(3) Platform: {platform}")

    img_frame_type = (
        dai.ImgFrame.Type.BGR888p
        if platform == dai.Platform.RVC2
        else dai.ImgFrame.Type.BGR888i
    )
    logger.debug(f"(3.1) Image frame type: {img_frame_type}")

    try:
        model_description = dai.NNModelDescription(
            "luxonis/yunet:640x360", platform.name
        )
        nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
        logger.debug("(4) NN archive created.")
    except Exception as e:
        logger.warning(e)
        logger.warning("Can't create NN archive for model: luxonis/yunet:640x360")
        exit(7)

    cam = pipeline.create(dai.node.Camera).build()
    logger.debug("(5) Camera node created.")

    cam_out = cam.requestOutput(IMG_SHAPE, type=img_frame_type, fps=FPS_LIMIT)
    logger.debug("(6) Camera output created.")

    tiling = pipeline.create(Tiling).build(
        overlap=0.1,
        img_output=cam_out,
        grid_size=(2, 2),
        img_shape=IMG_SHAPE,
        nn_shape=nn_archive.getInputSize(),
        resize_mode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
    )
    logger.debug("(7) Tiling node created.")
    nn = pipeline.create(ParsingNeuralNetwork).build(tiling.out, nn_archive)
    logger.debug("(8) ParsingNeuralNetwork node created.")
    patcher = pipeline.create(TilesPatcher).build(cam_out, nn.out)
    logger.debug("(9) TilesPatcher node created.")

    patcher_out = patcher.out.createOutputQueue()
    logger.debug("(10) Patcher output created.")

    logger.debug("(11) Pipeline created.")

    pipeline.start()
    logger.debug("(12) Pipeline started.")

    while pipeline.isRunning():
        pipeline.processTasks()
        patcher_out: dai.ImgDetections = patcher_out.get()
        logger.debug("(13) Patcher output received.")
        pipeline.stop()
        break
