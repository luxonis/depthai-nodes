import argparse
import pickle

import depthai as dai
from check_messages import test_output
from utils import extract_main_slug, extract_parser

from depthai_nodes.node import ParserGenerator

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "-m",
    "--model",
    help="The model from which the parser is built",
    required=True,
    type=str,
)
argparser.add_argument("-ip", help="IP address of the device", default="")

args = argparser.parse_args()
model_slug = args.model
IP_mxid = args.ip


def load_tensors(model: str, parser: str) -> dai.NNData:
    model = extract_main_slug(model)
    nn_data = dai.NNData()
    with open(f"nn_datas/{parser}/{model}.pkl", "rb") as f:
        data = pickle.load(f)
        for key, value in data.items():
            nn_data.addTensor(str(key), value.tolist())
        return nn_data


try:
    device = dai.Device(dai.DeviceInfo(IP_mxid))
except Exception as e:
    print(f"Error: {e}")
    exit(6)

device_platform = device.getPlatform().name

# Get the model from the HubAI
try:
    model_description = dai.NNModelDescription(model=model_slug, platform="RVC2")
    archive_path = dai.getModelFromZoo(model_description)
except Exception:
    try:
        model_description = dai.NNModelDescription(model=model_slug, platform="RVC4")
        archive_path = dai.getModelFromZoo(model_description)
    except Exception as e:
        print(f"Error: {e}")
        exit(7)

try:
    nn_archive = dai.NNArchive(archive_path)
except Exception as e:
    print(f"Error: {e}")
    exit(8)

try:
    parser_name = extract_parser(nn_archive)
except Exception as e:
    print(e)
    exit(9)


class Sender(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.out = self.createOutput()

    def run(self):
        try:
            nnData = load_tensors(model_slug, parser_name)
        except Exception as e:
            nnData = dai.NNData()
            print(e)
            return 10
        self.out.send(nnData)


with dai.Pipeline(device) as pipeline:
    parser = pipeline.create(ParserGenerator).build(nn_archive=nn_archive)[0]

    sender = pipeline.create(Sender)

    sender.out.link(parser.input)
    parser_queue = parser.out.createOutputQueue()

    pipeline.start()

    while pipeline.isRunning():
        message = parser_queue.get()
        test_output(message, model_slug, parser_name)
        pipeline.stop()
