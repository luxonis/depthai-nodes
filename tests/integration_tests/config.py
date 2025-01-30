parsers_slugs = {
    "ClassificationParser": [
        "luxonis/efficientnet-lite:lite0-224x224",
        "luxonis/emotion-recognition:260x260",
    ],
    "ClassificationSequenceParser": ["luxonis/paddle-text-recognition:320x48"],
    "EmbeddingsParser": ["luxonis/arcface:lfw-112x112"],
    "FastSAMParser": ["luxonis/fastsam-s:512x288"],
    "HRNetParser": ["luxonis/lite-hrnet:18-coco-192x256"],
    "ImageOutputParser": ["luxonis/dncnn3:320x240"],
    "KeypointParser": ["luxonis/mediapipe-face-landmarker:192x192"],
    "LaneDetectionParser": ["luxonis/ultra-fast-lane-detection:culane-800x288"],
    "MapOutputParser": ["luxonis/dm-count:sha-426x240"],
    "MPPalmDetectionParser": ["luxonis/mediapipe-palm-detection:192x192"],
    "MLSDParser": ["luxonis/m-lsd-tiny:512x512"],
    "PPTextDetectionParser": ["luxonis/paddle-text-detection:256x256"],
    "SCRFDParser": ["luxonis/scrfd-face-detection:10g-640x640"],
    "SegmentationParser": [
        "luxonis/mediapipe-selfie-segmentation:256x144",
        "luxonis/deeplab-v3-plus:256x256",
    ],
    "SuperAnimalParser": ["luxonis/superanimal-landmarker:256x256"],
    "YOLOExtendedParser": [
        "luxonis/yolov8-instance-segmentation-nano:coco-512x288",
        "luxonis/yolov8-nano-pose-estimation:coco-512x288",
    ],
    "YuNetParser": ["luxonis/yunet:320x240"],
    "RegressionParser": ["luxonis/gaze-estimation-adas:60x60"],
}
