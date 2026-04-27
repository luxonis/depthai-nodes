# ruff: noqa: E402
import sys
from pathlib import Path

# set PYTHONPATH to src
current_dir = Path(__file__)
src_path = current_dir.parent.parent.parent
sys.path.insert(0, src_path.absolute().as_posix())

import depthai as dai
from merge_img_detections import MergeImgDetections

from depthai_nodes.node import (
    CoordinatesMapper,
    ExtendedNeuralNetwork,
    FrameCropper,
    GatherData,
    ParsingNeuralNetwork,
)

print("Starting")
RGB_WIDTH, RGB_HEIGHT = 1280, 720
RGB_HIGH_RES_WIDTH, RGB_HIGH_RES_HEIGHT = 3840 // 4, 2160 // 4
PEOPLE_DETECTION_MODEL = "luxonis/scrfd-person-detection:25g-640x640"
FACE_DETECTION_MODEL = "luxonis/yunet:320x240"
DEVICE_MXID = ""
FPS = 8

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(DEVICE_MXID)) if DEVICE_MXID else dai.Device()
platform = device.getPlatform().name
frame_type = (
    dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
)

with dai.Pipeline(device) as pipeline:
    rgb = pipeline.create(dai.node.Camera).build(sensorFps=FPS)
    rgb_high_res_out = rgb.requestOutput(
        size=(RGB_HIGH_RES_WIDTH, RGB_HIGH_RES_HEIGHT),
        type=frame_type,
        fps=FPS,
    )
    # 1st stage
    people_detection = pipeline.create(ExtendedNeuralNetwork).build(
        inputImage=rgb,
        resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
        nnSource=PEOPLE_DETECTION_MODEL,
    )
    # 2nd stage
    face_cropper = (
        pipeline.create(FrameCropper)
        .fromImgDetections(
            inputImgDetections=people_detection.out,
            outputSize=(320, 240),
            resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
        )
        .build(
            inputImage=rgb_high_res_out,
        )
    )
    face_detections = pipeline.create(ParsingNeuralNetwork).build(
        input=face_cropper.out,
        nnSource=FACE_DETECTION_MODEL,
    )
    face_detections_collected = pipeline.create(GatherData).build(
        inputData=face_detections.out,
        inputReference=people_detection.out,
        cameraFps=FPS,
    )
    face_detections_remapped = pipeline.create(CoordinatesMapper).build(
        toTransformationInput=people_detection.passthrough,
        fromTransformationInput=face_detections_collected.out,
    )
    face_detections_merged = pipeline.create(MergeImgDetections).build(
        input=face_detections_remapped.out,
    )

    visualizer.addTopic(topicName="rgb", output=people_detection.passthrough, group="1")
    visualizer.addTopic(topicName="people", output=people_detection.out, group="1")
    visualizer.addTopic(topicName="faces", output=face_detections_merged.out, group="1")

    pipeline.enablePipelineDebugging(enable=True)
    pipeline.start()
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
