from .image_output import ImageOutputParser
from .keypoints import KeypointParser
from .mediapipe_hand_detection import MPHandDetectionParser
from .mediapipe_hand_landmarker import MPHandLandmarkParser
from .mlsd import MLSDParser
from .monocular_depth import MonocularDepthParser
from .scrfd import SCRFDParser
from .segmentation import SegmentationParser
from .superanimal_landmarker import SuperAnimalParser
from .xfeat import XFeatParser
from .yunet import YuNetParser

__all__ = [
    "ImageOutputParser",
    "MonocularDepthParser",
    "YuNetParser",
    "MPHandDetectionParser",
    "MPHandLandmarkParser",
    "SCRFDParser",
    "SegmentationParser",
    "SuperAnimalParser",
    "KeypointParser",
    "MLSDParser",
    "XFeatParser",
]
