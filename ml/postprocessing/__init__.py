from .zero_dce import ZeroDCEParser
from .dncnn3 import DnCNN3Parser
from .monocular_depth import MonocularDepthParser
from .yunet import YuNetParser
from .mediapipe_hand_detection import MPHandDetectionParser
from .mediapipe_hand_landmarker import MPHandLandmarkParser
from .scrfd import SCRFDParser
from .segmentation import SegmentationParser

__all__ = [
    'ZeroDCEParser', 
    'DnCNN3Parser',
    'MonocularDepthParser',
    'YuNetParser',
    'MPHandDetectionParser',
    'MPHandLandmarkParser',
    'SCRFDParser',
    'SegmentationParser',
    ]
