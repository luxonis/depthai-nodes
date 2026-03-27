import sys
import depthai as dai
from pathlib import Path

# set PYTHONPATH to src
current_dir = Path(__file__)
src_path = current_dir.parent.parent.parent
sys.path.insert(0, src_path.absolute().as_posix())

from merge_img_detections import MergeImgDetections
from depthai_nodes.node.coordinates_mapper import CoordinatesMapper
from depthai_nodes.node.frame_cropper import FrameCropper
from depthai_nodes.node.gather_data import GatherData
from depthai_nodes.node.img_detections_filter import ImgDetectionsFilter
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node.tiling import Tiling

print("Starting")
RGB_WIDTH, RGB_HEIGHT = 1920, 1080
FACE_DETECTION_MODEL = "luxonis/yunet:320x240"
DEVICE_MXID = ""
FPS = 5
GRID_SIZE = (2, 2)

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(DEVICE_MXID)) if DEVICE_MXID else dai.Device()
platform = device.getPlatform().name
frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p

with dai.Pipeline(device) as pipeline:
    rgb = pipeline.create(dai.node.Camera).build(sensorFps=FPS)
    rgb_out = rgb.requestOutput(size=(RGB_WIDTH, RGB_HEIGHT), fps=FPS, type=frame_type)
    tiling = pipeline.create(
        Tiling
    ).build(
        overlap=0.2,
        trigger=rgb_out,
        gridSize=GRID_SIZE,
        canvasShape=(RGB_WIDTH, RGB_HEIGHT),
        resizeShape=(320, 240),
        resizeMode=dai.ImageManipConfig.ResizeMode.STRETCH,

    )
    frame_cropper = pipeline.create(
        FrameCropper
    ).fromManipConfigs(
        inputManipConfigs=tiling.out,
    ).build(
        inputImage=rgb_out,
        outputSize=(320, 240),
        resizeMode=dai.ImageManipConfig.ResizeMode.STRETCH,
    )

    face_detection = pipeline.create(
        ParsingNeuralNetwork
    ).build(
        input=frame_cropper.out,
        nnSource=FACE_DETECTION_MODEL,
    )
    coordinates_mapper = pipeline.create(
        CoordinatesMapper
    ).build(
        toTransformationInput=rgb_out,
        fromTransformationInput=face_detection.out,
    )
    message_gatherer = pipeline.create(
        GatherData
    ).build(
        inputData=coordinates_mapper.out,
        inputReference=rgb_out,
        cameraFps=FPS,
        waitCountFn=lambda _: GRID_SIZE[0] * GRID_SIZE[1],
    )
    merged_detections = pipeline.create(
        MergeImgDetections
    ).build(
        input=message_gatherer.out,
    )
    filter_node = pipeline.create(
        ImgDetectionsFilter
    ).useNms(
        confThresh=0.8,
        iouThresh=0.3,
    ).build(
        input=merged_detections.out,
    )

    visualizer.addTopic(topicName="rgb", output=rgb_out, group="1")
    visualizer.addTopic(topicName="dets", output=filter_node.out, group="1")

    pipeline.start()
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
