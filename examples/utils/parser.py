import depthai as dai

from depthai_nodes.ml.parsers import (
    ClassificationParser,
    FastSAMParser,
    KeypointParser,
    LaneDetectionParser,
    MapOutputParser,
    MPPalmDetectionParser,
    MultiClassificationParser,
    PaddleOCRParser,
    SCRFDParser,
    SegmentationParser,
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


def setup_map_output_parser(parser: MapOutputParser, params: dict):
    """Setup the map output parser with the required metadata."""
    try:
        min_max_scaling = params["min_max_scaling"]
        if min_max_scaling:
            parser.setMinMaxScaling(True)

    except Exception:
        print(
            "This NN archive does not have required metadata for MapOutputParser. Skipping setup..."
        )


def setup_land_detection_parser(parser: LaneDetectionParser, params: dict):
    """Setup the Lane Detection parser with the required metadata."""
    try:
        row_ancors = params["row_anchors"]
        griding_num = params["griding_num"]
        cls_num_per_lane = params["cls_num_per_lane"]
        parser.setRowAnchors(row_ancors)
        parser.setGridingNum(griding_num)
        parser.setClsNumPerLane(cls_num_per_lane)
    except Exception:
        print(
            "This NN archive does not have required metadata for LaneDetectionParser. Skipping setup..."
        )


def setup_fastsam_parser(parser: FastSAMParser, params: dict):
    """Setup the FastSAM parser with the required metadata."""
    try:
        conf_threshold = params["conf_threshold"]
        n_classes = params["n_classes"]
        iou_threshold = params["iou_threshold"]
        parser.setConfidenceThreshold(conf_threshold)
        parser.setIouThreshold(iou_threshold)
        parser.setNumClasses(n_classes)
        parser.setInputImageSize(512, 288)
        parser.setPrompt("everything")
    except Exception:
        print(
            "This NN archive does not have required metadata for FastSAMParser. Skipping setup..."
        )


def setup_paddleocr_parser(parser: PaddleOCRParser, params: dict):
    """Setup the PaddleOCR parser with the required metadata."""
    try:
        classes = params["classes"]
        parser.setClasses(classes)
    except Exception:
        print(
            "This NN archive does not have required metadata for PaddleOCRParser. Skipping setup..."
        )


def setup_multi_classification_parser(parser: MultiClassificationParser, params: dict):
    """Setup the Multi Classification parser with the required metadata."""
    try:
        classification_attributes = params["classification_attributes"]
        classification_labels = params["classification_labels"]
        parser.setClassificationAttributes(classification_attributes)
        parser.setClassificationLabels(classification_labels)
    except Exception:
        print(
            "This NN archive does not have required metadata for MultiClassificationParser. Skipping setup..."
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
    elif parser_name == "MapOutputParser":
        setup_map_output_parser(parser, extraParams)
    elif parser_name == "YOLOExtendedParser":
        setup_yolo_extended_parser(parser, extraParams)
    elif parser_name == "MPPalmDetectionParser":
        setup_palm_detection_parser(parser, extraParams)
    elif parser_name == "LaneDetectionParser":
        setup_land_detection_parser(parser, extraParams)
    elif parser_name == "FastSAMParser":
        setup_fastsam_parser(parser, extraParams)
    elif parser_name == "PaddleOCRParser":
        setup_paddleocr_parser(parser, extraParams)
    elif parser_name == "MultiClassificationParser":
        setup_multi_classification_parser(parser, extraParams)
