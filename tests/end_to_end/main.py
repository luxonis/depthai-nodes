import argparse
from typing import Tuple

import depthai as dai

from depthai_nodes.parsing_neural_network import ParsingNeuralNetwork


def parse_model_slug(full_slug) -> Tuple[str, str]:
    if ":" not in full_slug:
        raise NameError(
            "Please provide the model slug in the format of 'model_slug:model_version_slug'"
        )
    model_slug_parts = full_slug.split(":")
    model_slug = model_slug_parts[0]
    model_version_slug = model_slug_parts[1]

    return model_slug, model_version_slug


parser = argparse.ArgumentParser()
parser.add_argument(
    "-nn", "--nn_archive", type=str, default=None, help="Path to the NNArchive."
)
parser.add_argument(
    "-s", "--model_slug", type=str, default=None, help="Slug of the model from HubAI."
)
parser.add_argument("-ip", type=str, default="", help="IP of the device")
args = parser.parse_args()

if not (args.nn_archive or args.model_slug):
    raise ValueError("You have to pass either path to NNArchive or model slug")

device = dai.Device(dai.DeviceInfo(args.ip))
with dai.Pipeline(device) as pipeline:
    camera_node = pipeline.create(dai.node.Camera).build()

    if args.model_slug:
        model_slug, model_version_slug = parse_model_slug(args.model_slug)
        model = dai.NNModelDescription(
            modelSlug=model_slug, modelVersionSlug=model_version_slug
        )

    else:
        model = dai.NNArchive(args.nn_archive)

    nn_w_parser = pipeline.create(ParsingNeuralNetwork).build(camera_node, model)

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
