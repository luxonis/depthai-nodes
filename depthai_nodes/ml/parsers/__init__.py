from .age_gender import AgeGenderParser
from .classification import ClassificationParser
from .fastsam import FastSAMParser
from .hrnet import HRNetParser
from .image_output import ImageOutputParser
from .keypoints import KeypointParser
from .mediapipe_hand_landmarker import MPHandLandmarkParser
from .mediapipe_palm_detection import MPPalmDetectionParser
from .mlsd import MLSDParser
from .monocular_depth import MonocularDepthParser
from .scrfd import SCRFDParser
from .segmentation import SegmentationParser
from .superanimal_landmarker import SuperAnimalParser
from .thermal_image import ThermalImageParser
from .xfeat import XFeatParser
from .yolo import YOLOExtendedParser
from .yunet import YuNetParser

__all__ = [
    "ImageOutputParser",
    "MonocularDepthParser",
    "YuNetParser",
    "MPPalmDetectionParser",
    "MPHandLandmarkParser",
    "SCRFDParser",
    "SegmentationParser",
    "SuperAnimalParser",
    "KeypointParser",
    "MLSDParser",
    "XFeatParser",
    "ThermalImageParser",
    "ClassificationParser",
    "YOLOExtendedParser",
    "FastSAMParser",
    "AgeGenderParser",
    "HRNetParser",
]
