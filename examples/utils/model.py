from typing import List, Tuple

import depthai as dai

from depthai_nodes.ml.parsers import (
    ClassificationParser,
    KeypointParser,
    MonocularDepthParser,
    SCRFDParser,
    SegmentationParser,
    XFeatParser,
)


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


def setup_scrfd_parser(parser: SCRFDParser, params: dict):
    """Setup the SCRFD parser with the required metadata."""
    try:
        num_anchors = params["num_anchors"]
        feat_stride_fpn = params["feat_stride_fpn"]
        parser.setNumAnchors(num_anchors)
        parser.setFeatStrideFPN(feat_stride_fpn)
    except Exception:
        print(
            "This NN archive does not have required metadata for SCRFDParser. Skipping setup..."
        )


def setup_segmentation_parser(parser: SegmentationParser, params: dict):
    """Setup the segmentation parser with the required metadata."""
    try:
        background_class = params["background_class"]
        parser.setBackgroundClass(background_class)
    except Exception:
        print(
            "This NN archive does not have required metadata for SegmentationParser. Skipping setup..."
        )


def setup_keypoint_parser(parser: KeypointParser, params: dict):
    """Setup the keypoint parser with the required metadata."""
    try:
        num_keypoints = params["n_keypoints"]
        scale_factor = params["scale_factor"]
        parser.setNumKeypoints(num_keypoints)
        parser.setScaleFactor(scale_factor)
    except Exception:
        print(
            "This NN archive does not have required metadata for KeypointParser. Skipping setup..."
        )


def setup_classification_parser(parser: ClassificationParser, params: dict):
    """Setup the classification parser with the required metadata."""
    try:
        classes = params["classes"]
        is_softmax = params["is_softmax"]
        parser.setClasses(classes)
        parser.setSoftmax(is_softmax)
    except Exception:
        print(
            "This NN archive does not have required metadata for ClassificationParser. Skipping setup..."
        )


def setup_monocular_depth_parser(parser: MonocularDepthParser, params: dict):
    """Setup the monocular depth parser with the required metadata."""
    try:
        depth_type = params["depth_type"]
        if depth_type == "relative":
            parser.setRelativeDepthType()
        else:
            parser.setMetricDepthType()
    except Exception:
        print(
            "This NN archive does not have required metadata for MonocularDepthParser. Skipping setup..."
        )


def setup_xfeat_parser(parser: XFeatParser, params: dict):
    """Setup the XFeat parser with the required metadata."""
    try:
        input_size = params["input_size"]
        parser.setInputSize(input_size)
        parser.setOriginalSize(input_size)
    except Exception:
        print(
            "This NN archive does not have required metadata for XFeatParser. Skipping setup..."
        )


def setup_parser(parser: dai.ThreadedNode, nn_archive: dai.NNArchive, parser_name: str):
    """Setup the parser with the NN archive."""

    extraParams = (
        nn_archive.getConfig().getConfigV1().model.heads[0].metadata.extraParams
    )

    if parser_name == "SCRFDParser":
        setup_scrfd_parser(parser, extraParams)
    elif parser_name == "SegmentationParser":
        setup_segmentation_parser(parser, extraParams)
    elif parser_name == "KeypointParser":
        setup_keypoint_parser(parser, extraParams)
    elif parser_name == "ClassificationParser":
        setup_classification_parser(parser, extraParams)
    elif parser_name == "MonocularDepthParser":
        setup_monocular_depth_parser(parser, extraParams)
    elif parser_name == "XFeatParser":
        setup_xfeat_parser(parser, extraParams)
