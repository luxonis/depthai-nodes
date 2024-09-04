import depthai as dai

from depthai_nodes.ml.parsers import (
    ClassificationParser,
    KeypointParser,
    MonocularDepthParser,
    MPPalmDetectionParser,
    SCRFDParser,
    SegmentationParser,
    XFeatParser,
    YOLOExtendedParser,
)


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
        depth_limit = params["depth_limit"]
        if depth_type == "relative":
            parser.setRelativeDepthType()
        else:
            parser.setMetricDepthType()
        parser.setDepthLimit(depth_limit)
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


def setup_yolo_extended_parser(parser: YOLOExtendedParser, params: dict):
    """Setup the YOLO parser with the required metadata."""
    try:
        n_classes = params["n_classes"]
        parser.setNumClasses(n_classes)
    except Exception:
        print(
            "This NN archive does not have required metadata for YOLOExtendedParser. Skipping setup..."
        )


def setup_palm_detection_parser(parser: MPPalmDetectionParser, params: dict):
    """Setup the Palm Detection parser with the required metadata."""
    try:
        scale = params["scale"]
        parser.setScale(scale)
    except Exception:
        print(
            "This NN archive does not have required metadata for MPPalmDetectionParser. Skipping setup..."
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
    elif parser_name == "YOLOExtendedParser":
        setup_yolo_extended_parser(parser, extraParams)
    elif parser_name == "MPPalmDetectionParser":
        setup_palm_detection_parser(parser, extraParams)
