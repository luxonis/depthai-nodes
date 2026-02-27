import argparse

import depthai as dai

from depthai_nodes.node.parsers import RFDETRParser

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nn", "--nn_path", help="blob model path", required=True, type=str
)  # beware not to use .superblob - not supported yet!
parser.add_argument("-d", "--device", help="device to use", required=True, type=str)
args = parser.parse_args()

nn_path = args.nn_path
device = args.device

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()
print(f"Platform: {platform}")

nn_archive = dai.NNArchive(nn_path)


with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    detection_nn = pipeline.create(dai.node.NeuralNetwork).build(
        cam.requestOutput((384, 384), type=dai.ImgFrame.Type.BGR888i), nn_archive
    )
    parser = RFDETRParser()
    detection_nn.out.link(parser.input)

    visualizer.addTopic("Video", detection_nn.passthrough, "images")
    visualizer.addTopic("Detections", parser.out, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
