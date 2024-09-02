from typing import List, Tuple

import depthai as dai

from depthai_nodes.ml.parsers import *


def get_model_from_hub(model_slug: str, model_version_slug: str) -> dai.NNArchive:
    """Get the model from the HubAI and return the NN archive."""
    print(
        f"Downloading model {model_slug} with version {model_version_slug} from HubAI..."
    )
    modelDescription = dai.NNModelDescription(
        modelSlug=model_slug, modelVersionSlug=model_version_slug, platform="RVC2"
    )
    archivePath = dai.getModelFromZoo(modelDescription)
    print("Download successful!")
    nn_archive = dai.NNArchive(archivePath)

    return nn_archive


def get_parser_from_archive(nn_archive: dai.NNArchive) -> str:
    """Get the required parser from the NN archive."""
    try:
        required_parser = nn_archive.getConfig().getConfigV1().model.heads[0].parser
    except AttributeError:
        print(
            "This NN archive does not have a parser. Please use NN archives that have parsers."
        )
        exit(1)

    print(f"Required parser: {required_parser}")

    return required_parser


def get_parser(nn_archive: dai.NNArchive) -> Tuple[dai.ThreadedNode, str]:
    """Map the parser from the NN archive to the actual parser in depthai-nodes."""
    required_parser = get_parser_from_archive(nn_archive)

    if required_parser == "YOLO" or required_parser == "SSD":
        return None, required_parser

    parser = globals().get(required_parser, None)

    if parser is None:
        raise NameError(
            f"Parser {required_parser} is not available in the depthai_nodes.ml.parsers module."
        )

    return parser, required_parser


def get_inputs_from_archive(nn_archive: dai.NNArchive) -> List:
    """Get all inputs from NN archive."""
    try:
        inputs = nn_archive.getConfig().getConfigV1().model.inputs
    except AttributeError:
        print(
            "This NN archive does not have an input shape. Please use NN archives that have input shapes."
        )
        exit(1)

    return inputs


def get_input_shape(nn_archive: dai.NNArchive) -> Tuple[int, int]:
    """Get the input shape of the model from the NN archive."""
    inputs = get_inputs_from_archive(nn_archive)

    if len(inputs) > 1:
        raise ValueError(
            "This model has more than one input. Currently, only models with one input are supported."
        )

    try:
        input_shape = inputs[0].shape[2:][::-1]
    except AttributeError:
        print(
            "This NN archive does not have an input shape. Please use NN archives that have input shapes."
        )
        exit(1)

    print(f"Input shape: {input_shape}")

    return input_shape
