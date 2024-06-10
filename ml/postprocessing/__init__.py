from .image_output import ImageOutputParser
from .monocular_depth import MonocularDepthParser
from .yunet import YuNetParser
from .mediapipe_hand_detection import MPHandDetectionParser
from .mediapipe_hand_landmarker import MPHandLandmarkParser
from .scrfd import SCRFDParser
from .segmentation import SegmentationParser
from .superanimal_landmarker import SuperAnimalParser
from .mediapipe_face_landmarker import MPFaceLandmarkerParser

__all__ = [
    'ImageOutputParser',
    'MonocularDepthParser',
    'YuNetParser',
    'MPHandDetectionParser',
    'MPHandLandmarkParser',
    'SCRFDParser',
    'SegmentationParser',
    'SuperAnimalParser',
    'MPFaceLandmarkerParser',
    ]
