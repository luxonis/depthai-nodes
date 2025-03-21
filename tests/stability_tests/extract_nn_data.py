import argparse
import os
import pickle

import cv2
import depthai as dai
from utils import extract_main_slug, extract_parser, get_input_shape

argparser = argparse.ArgumentParser()
argparser.add_argument("-ip", help="IP address of the device", default="")
argparser.add_argument(
    "-img", "--img_path", help="Path to the input image", required=True, type=str
)
argparser.add_argument(
    "-m", "--model", help="Model from HubAI", required=True, type=str
)

args = argparser.parse_args()
IP_mxid = args.ip
img_path = args.img_path
model: str = args.model

device = dai.Device(dai.DeviceInfo(IP_mxid))
device_platform = device.getPlatform().name

# Get the model from the HubAI
model_description = dai.NNModelDescription(model=model, platform=device_platform)
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

input_size = get_input_shape(nn_archive)
input_width = input_size[0]
input_height = input_size[1]
parser = extract_parser(nn_archive)


class HostImage(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.out = self.createOutput()

    def run(self):
        frame = cv2.imread(img_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image from {img_path}")
        else:
            frame = cv2.resize(frame, (input_width, input_height))
            if device_platform == "RVC2":
                frame = frame.transpose(2, 0, 1)

            imgFrame = dai.ImgFrame()

            imgFrame.setFrame(frame)
            imgFrame.setWidth(input_width)
            imgFrame.setHeight(input_height)
            imgFrame.setType(
                dai.ImgFrame.Type.BGR888p
                if device_platform == "RVC2"
                else dai.ImgFrame.Type.BGR888i
            )
            print("input frame HW:", imgFrame.getHeight(), imgFrame.getWidth())

            self.out.send(imgFrame)


with dai.Pipeline(device) as pipeline:
    image_node = pipeline.create(HostImage)

    model_platforms = [platform.name for platform in nn_archive.getSupportedPlatforms()]

    if device_platform not in model_platforms:
        print(f"Model not supported on {device_platform}.")
        device.close()
        exit(5)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setNNArchive(nn_archive)
    image_node.out.link(detection_nn.input)

    nn_queue = detection_nn.out.createOutputQueue()
    pass_queue = detection_nn.passthrough.createOutputQueue()

    pipeline.start()

    while pipeline.isRunning():
        nn_data: dai.NNData = nn_queue.get()
        frame: dai.ImgFrame = pass_queue.get().getCvFrame()
        tensor_names = nn_data.getAllLayerNames()
        tensors = {}
        print("-------------------------------")
        print("Tensor name | Shape | Data type")
        print("-------------------------------")
        for tensor_name in tensor_names:
            tensor = nn_data.getTensor(tensor_name)
            print(tensor_name, tensor.shape, tensor.dtype)
            tensors[tensor_name] = tensor

        if not os.path.exists("nn_datas"):
            os.makedirs("nn_datas")
        if not os.path.exists(f"nn_datas/{parser}"):
            os.makedirs(f"nn_datas/{parser}")
        with open(f"nn_datas/{parser}/{extract_main_slug(model)}.pkl", "wb") as f:
            pickle.dump(tensors, f)
        cv2.imwrite(f"nn_datas/{parser}/{extract_main_slug(model)}.png", frame)

        pipeline.stop()

device.close()
