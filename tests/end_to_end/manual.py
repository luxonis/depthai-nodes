import argparse

import depthai as dai
from utils import get_input_shape, get_num_inputs, parse_model_slug

from depthai_nodes.parsing_neural_network import ParsingNeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nn", "--nn_archive", type=str, default=None, help="Path to the NNArchive."
)
parser.add_argument(
    "-s", "--model_slug", type=str, default=None, help="Slug of the model from HubAI."
)
parser.add_argument("-ip", type=str, default="", help="IP of the device")
args = parser.parse_args()

if args.model_slug:
    if "xfeat" in args.model_slug:
        print("XFeat model is not supported in this test.")
        exit(8)

if not (args.nn_archive or args.model_slug):
    raise ValueError("You have to pass either path to NNArchive or model slug")

try:
    device = dai.Device(dai.DeviceInfo(args.ip))
except Exception as e:
    print(e)
    print("Can't connect to the device with IP/mxid: ", args.ip)
    exit(6)

with dai.Pipeline(device) as pipeline:
    camera_node = pipeline.create(dai.node.Camera).build()

    if args.model_slug:
        model_slug, model_version_slug = parse_model_slug(args.model_slug)
        model_desc = dai.NNModelDescription(
            modelSlug=model_slug,
            modelVersionSlug=model_version_slug,
            platform=device.getPlatform().name,
        )
        try:
            nn_archive_path = dai.getModelFromZoo(model_desc, useCached=False)
        except Exception as e:
            print(e)
            print(
                f"Couldn't find model {args.model_slug} for {device.getPlatform().name} in the ZOO"
            )
            device.close()
            exit(7)
        try:
            nn_archive = dai.NNArchive(nn_archive_path)
        except Exception as e:
            print(e)
            print(f"Couldn't load the model {args.model_slug} from NN archive.")
            device.close()
            exit(9)

    else:
        nn_archive = dai.NNArchive(args.nn_archive)

    model_platforms = [platform.name for platform in nn_archive.getSupportedPlatforms()]

    if device.getPlatform().name not in model_platforms:
        print(f"Model not supported on {device.getPlatform().name}.")
        device.close()
        exit(5)

    if get_num_inputs(nn_archive) > 1:
        print(
            "This model has more than one input. Currently, only models with one input are supported."
        )
        device.close()
        exit(8)

    try:
        input_size = get_input_shape(nn_archive)
    except Exception as e:
        print(e)
        device.close()
        exit(8)

    if input_size[0] % 2 != 0 or input_size[1] % 2 != 0:
        print(
            f"We only support even numbers for resolution sizes. Got {input_size[0]}x{input_size[1]}."
        )
        device.close()
        exit(8)

    if input_size[0] < 128 and input_size[1] < 128:
        print("Input size is too small for the device.")
        device.close()
        exit(8)

    nn_w_parser = pipeline.create(ParsingNeuralNetwork).build(camera_node, nn_archive)

    head_indices = nn_w_parser._parsers.keys()

    parser_output_queues = {
        i: nn_w_parser.getOutput(i).createOutputQueue() for i in head_indices
    }

    pipeline.start()

    while pipeline.isRunning():
        for head_id in parser_output_queues:
            parser_output = parser_output_queues[head_id].get()
            print(f"{head_id} - {type(parser_output)}")
        pipeline.stop()

device.close()
