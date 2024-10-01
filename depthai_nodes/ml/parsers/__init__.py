from .age_gender import AgeGenderParser
from .classification import ClassificationParser, MultiClassificationParser
from .fastsam import FastSAMParser
from .hrnet import HRNetParser
from .image_output import ImageOutputParser
from .keypoints import KeypointParser
from .lane_detection import LaneDetectionParser
from .map_output import MapOutputParser
from .mediapipe_hand_landmarker import MPHandLandmarkParser
from .mediapipe_palm_detection import MPPalmDetectionParser
from .mlsd import MLSDParser
from .ppdet import PPTextDetectionParser
from .ppocr import PaddleOCRParser
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
    "AgeGenderParser",
    "HRNetParser",
    "PPTextDetectionParser",
    "MapOutputParser",
    "PaddleOCRParser",
    "LaneDetectionParser",
    "MultiClassificationParser",
]
