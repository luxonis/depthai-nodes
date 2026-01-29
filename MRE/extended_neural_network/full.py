import sys
import depthai as dai
from pathlib import Path

from depthai_nodes.node.extended_neural_network import ExtendedNeuralNetwork

# set PYTHONPATH to src
current_dir = Path(__file__)
src_path = current_dir.parent.parent.parent.parent
sys.path.insert(0, src_path.absolute().as_posix())

print("Starting")
LOW_RES_WIDTH, LOW_RES_HEIGHT = 800, 800
PEOPLE_DETECTION_MODEL = "luxonis/scrfd-person-detection:25g-640x640"
DEVICE_MXID = ""
FPS = 15

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(DEVICE_MXID)) if DEVICE_MXID else dai.Device()
platform = device.getPlatform().name
frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p

with dai.Pipeline(device) as pipeline:
    rgb = pipeline.create(dai.node.Camera).build(sensorFps=FPS)
    rgb_out = rgb.requestOutput(size=(LOW_RES_WIDTH, LOW_RES_HEIGHT), fps=FPS, type=frame_type)
    rgb_for_encoder = rgb.requestOutput(size=(LOW_RES_WIDTH, LOW_RES_HEIGHT), fps=FPS, type=dai.ImgFrame.Type.NV12)
    encoder = pipeline.create(dai.node.VideoEncoder).build(
        profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        input=rgb_for_encoder,
    )

    people_detection = pipeline.create(ExtendedNeuralNetwork).with_tiling(
        image_input_shape=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
        grid_size=(2, 2),
        global_detection=True,
    ).with_detections_filter(
        confidence_threshold=0.75,
        labels_to_keep=[0],
    ).build(
        image_input=rgb_out,
        input_resize_mode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
        nn_source=PEOPLE_DETECTION_MODEL,
    )
    visualizer.addTopic(topicName="rgb", output=encoder.out)
    visualizer.addTopic(topicName="detections", output=people_detection.out)

    pipeline.start()
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
