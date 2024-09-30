from .base_parser import BaseParser
from .classification import ClassificationParser
from .fastsam import FastSAMParser
from .hrnet import HRNetParser
from .image_output import ImageOutputParser
from .keypoints import KeypointParser
from .lane_detection import LaneDetectionParser
from .map_output import MapOutputParser
from .mediapipe_hand_landmarker import MPHandLandmarkParser
from .mediapipe_palm_detection import MPPalmDetectionParser
from .mlsd import MLSDParser
from .parser_generator import ParserGenerator
from .ppdet import PPTextDetectionParser
from .ppocr import PaddleOCRParser
from .regression import RegressionParser
from .scrfd import SCRFDParser
from .segmentation import SegmentationParser
from .superanimal_landmarker import SuperAnimalParser
from .xfeat_mono import XFeatMonoParser
from .xfeat_stereo import XFeatStereoParser
from .yolo import YOLOExtendedParser
from .yunet import YuNetParser

__all__ = [
    "ImageOutputParser",
    "YuNetParser",
    "MPPalmDetectionParser",
    "MPHandLandmarkParser",
    "SCRFDParser",
    "SegmentationParser",
    "SuperAnimalParser",
    "KeypointParser",
    "MLSDParser",
    "XFeatMonoParser",
    "XFeatStereoParser",
    "ClassificationParser",
    "YOLOExtendedParser",
    "FastSAMParser",
    "RegressionParser",
    "HRNetParser",
    "PPTextDetectionParser",
    "MapOutputParser",
    "PaddleOCRParser",
    "LaneDetectionParser",
    "ParserGenerator",
    "BaseParser",
]
