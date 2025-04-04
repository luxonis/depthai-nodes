SLUGS = [
    "yunet:new-240x320",
    "mediapipe-hand-landmarker:224x224",
    "age-gender-recognition:new-62x62",
    "paddle-text-recognition:160x48",
    "midas-v2-1:small-192x256",
    "paddle-text-detection:256x256",
    "deeplab-v3-plus:256x256",
    "yolov8-nano-pose-estimation:coco-512x288",
    "yolov8-instance-segmentation-nano:coco-512x288",
    "ewasr:384x512",
    "m-lsd:512x512",
]
# SLUGS = [
#     "yunet:new-new-480x640",
#     "osnet:market1501-128x256",
#     "osnet:imagenet-128x256",
#     "osnet:multi-source-domain-128x256",
#     "arcface:lfw-112x112",
#     "mediapipe-hand-landmarker:224x224",
#     "mediapipe-hand-landmarker:old-224x224",
#     "head-pose-estimation:60x60",
#     "yunet:new-960x1280",
#     "age-gender-recognition:new-62x62",
#     "yunet:new-240x320",
#     "yunet:new-480x640",
#     "yunet:new-720x960",
#     "yunet:320x320",
#     "yunet:640x640",
#     "lite-hrnet:30-coco-384x288",
#     "lite-hrnet:30-coco-256x192",
#     "lite-hrnet:18-coco-384x288",
#     "lite-hrnet:18-coco-256x192",
#     "xfeat:stereo-480x640",
#     "xfeat:mono-480x640",
#     "xfeat:stereo-240x320",
#     "xfeat:mono-240x320",
#     "objectron:camera-224x224",
#     "objectron:chair-224x224",
#     "objectron:cup-224x224",
#     "objectron:sneakers-224x224",
#     "xfeat:240x320",
#     "xfeat:480x640",
#     "depth-anything-v2:vit-s-mde-outdoors-252x336",
#     "depth-anything-v2:vit-s-mde-indoors-252x336",
#     "depth-anything-v2:vit-s-420x560",
#     "depth-anything-v2:vit-s-252x336",
#     "paddle-text-recognition:160x48",
#     "midas-v2-1:small-768x1024",
#     "midas-v2-1:small-384x512",
#     "midas-v2-1:small-192x256",
#     "dm-count:sha-144x256",
#     "dm-count:sha-240x426",
#     "dm-count:sha-360x640",
#     "dm-count:sha-540x960",
#     "dm-count:shb-144x256",
#     "dm-count:shb-240x426",
#     "dm-count:shb-360x640",
#     "dm-count:shb-540x960",
#     "dm-count:qnrf-144x256",
#     "dm-count:qnrf-240x426",
#     "dm-count:qnrf-360x640",
#     "dm-count:qnrf-540x960",
#     "ppe-detection:640x640",
#     "midas-v2-1:small-288x512",
#     "midas-v2-1:small-256x384",
#     "paddle-text-recognition:320x48",
#     "paddle-text-detection:256x256",
#     "paddle-text-detection:544x960",
#     "paddle-text-detection:320x576",
#     "ultra-fast-lane-detection:culane-800x288",
#     "ultra-fast-lane-detection:tusimple-800x288",
#     "deeplab-v3-plus:256x256",
#     "deeplab-v3-plus:person-513x513",
#     "deeplab-v3-plus:person-256x256",
#     "deeplab-v3-plus:513x513",
#     "yolov6-nano:coco-416x416",
#     "pp-liteseg:512x1024",
#     "yolov8-large-pose-estimation:coco-640x352",
#     "yolov8-nano-pose-estimation:coco-512x288",
#     "yolov8-instance-segmentation-nano:coco-512x288",
#     "yolov8-instance-segmentation-large:coco-640x352",
#     "fastsam-x:640x352",
#     "fastsam-s:512x288",
#     "yolov10-nano:coco-512x288",
#     "ewasr:384x512",
#     "m-lsd:512x512",
#     "m-lsd-tiny:512x512",
#     "fastsam-s:clip-visual",
#     "scrfd-person-detection:2-5g-640x640",
#     "emotion-recognition:gray-64x64",
#     "efficientnet-lite:lite4-300x300",
#     "efficientnet-lite:lite0-224x224",
#     "mobilenet-ssd:300x300",
#     "license-plate-detection:640x640",
#     "vehicle-attributes-classification:72x72",
#     "emotion-recognition:260x260",
#     "age-gender-recognition:62x62",
#     "fastsam-s:clip-textual",
#     "mediapipe-palm-detection:128x128",
#     "xfeat:352x640",
#     "esrgan:256x256",
#     "rt-super-resolution:50x50",
#     "qrdet:nano-288x512",
#     "yolov6-large:r2-coco-640x352",
#     "yolov6-nano:r2-adjusted-coco-512x288",
#     "yolov6-nano:r2-coco-512x288",
#     "zero-dce:400x600",
#     "dncnn3:321x481",
#     "mediapipe-selfie-segmentation:144x256",
#     "superanimal-landmarker:256x256",
#     "mediapipe-palm-detection:192x192",
#     "mediapipe-face-landmarker:192x192",
#     "scrfd-face-detection:10g-640x640",
# ]

PARSERS_SLUGS = {
    "YuNetParser": [
        "yunet:new-new-480x640",
        "yunet:new-960x1280",
        "yunet:new-240x320",
        "yunet:new-480x640",
        "yunet:new-720x960",
        "yunet:320x320",
        "yunet:640x640",
    ],
    "EmbeddingsParser": [
        "osnet:market1501-128x256",
        "osnet:imagenet-128x256",
        "osnet:multi-source-domain-128x256",
        "arcface:lfw-112x112",
    ],
    "KeypointParser": [
        "mediapipe-hand-landmarker:224x224",
        "objectron:camera-224x224",
        "objectron:chair-224x224",
        "objectron:cup-224x224",
        "objectron:sneakers-224x224",
        "mediapipe-face-landmarker:192x192",
    ],
    "RegressionParser": [
        "mediapipe-hand-landmarker:224x224",
        "mediapipe-hand-landmarker:224x224",
        "head-pose-estimation:60x60",
        "head-pose-estimation:60x60",
        "head-pose-estimation:60x60",
        "age-gender-recognition:new-62x62",
        "objectron:camera-224x224",
        "objectron:chair-224x224",
        "objectron:cup-224x224",
        "objectron:sneakers-224x224",
    ],
    "MPHandLandmarkParser": ["mediapipe-hand-landmarker:old-224x224"],
    "ClassificationParser": [
        "age-gender-recognition:new-62x62",
        "emotion-recognition:gray-64x64",
        "efficientnet-lite:lite4-300x300",
        "efficientnet-lite:lite0-224x224",
        "vehicle-attributes-classification:72x72",
        "vehicle-attributes-classification:72x72",
        "emotion-recognition:260x260",
    ],
    "HRNetParser": [
        "lite-hrnet:30-coco-384x288",
        "lite-hrnet:30-coco-256x192",
        "lite-hrnet:18-coco-384x288",
        "lite-hrnet:18-coco-256x192",
    ],
    "XFeatStereoParser": ["xfeat:stereo-480x640", "xfeat:stereo-240x320"],
    "XFeatMonoParser": ["xfeat:mono-480x640", "xfeat:mono-240x320"],
    "XFeatParser": ["xfeat:240x320", "xfeat:480x640", "xfeat:352x640"],
    "MapOutputParser": [
        "depth-anything-v2:vit-s-mde-outdoors-252x336",
        "depth-anything-v2:vit-s-mde-indoors-252x336",
        "depth-anything-v2:vit-s-420x560",
        "depth-anything-v2:vit-s-252x336",
        "midas-v2-1:small-768x1024",
        "midas-v2-1:small-384x512",
        "midas-v2-1:small-192x256",
        "dm-count:sha-144x256",
        "dm-count:sha-240x426",
        "dm-count:sha-360x640",
        "dm-count:sha-540x960",
        "dm-count:shb-144x256",
        "dm-count:shb-240x426",
        "dm-count:shb-360x640",
        "dm-count:shb-540x960",
        "dm-count:qnrf-144x256",
        "dm-count:qnrf-240x426",
        "dm-count:qnrf-360x640",
        "dm-count:qnrf-540x960",
        "midas-v2-1:small-288x512",
        "midas-v2-1:small-256x384",
    ],
    "ClassificationSequenceParser": [
        "paddle-text-recognition:160x48",
        "paddle-text-recognition:320x48",
    ],
    "YOLO": [
        "ppe-detection:640x640",
        "yolov6-nano:coco-416x416",
        "yolov10-nano:coco-512x288",
        "license-plate-detection:640x640",
        "qrdet:nano-288x512",
        "yolov6-large:r2-coco-640x352",
        "yolov6-nano:r2-adjusted-coco-512x288",
        "yolov6-nano:r2-coco-512x288",
    ],
    "PPTextDetectionParser": [
        "paddle-text-detection:256x256",
        "paddle-text-detection:544x960",
        "paddle-text-detection:320x576",
    ],
    "LaneDetectionParser": [
        "ultra-fast-lane-detection:culane-800x288",
        "ultra-fast-lane-detection:tusimple-800x288",
    ],
    "SegmentationParser": [
        "deeplab-v3-plus:256x256",
        "deeplab-v3-plus:person-513x513",
        "deeplab-v3-plus:person-256x256",
        "deeplab-v3-plus:513x513",
        "pp-liteseg:512x1024",
        "ewasr:384x512",
        "mediapipe-selfie-segmentation:144x256",
    ],
    "YOLOExtendedParser": [
        "yolov8-large-pose-estimation:coco-640x352",
        "yolov8-nano-pose-estimation:coco-512x288",
        "yolov8-instance-segmentation-nano:coco-512x288",
        "yolov8-instance-segmentation-large:coco-640x352",
    ],
    "FastSAMParser": ["fastsam-x:640x352", "fastsam-s:512x288"],
    "MLSDParser": ["m-lsd:512x512", "m-lsd-tiny:512x512"],
    "SCRFDParser": [
        "scrfd-person-detection:2-5g-640x640",
        "scrfd-face-detection:10g-640x640",
    ],
    "SSD": ["mobilenet-ssd:300x300"],
    "AgeGenderParser": ["age-gender-recognition:62x62"],
    "MPPalmDetectionParser": [
        "mediapipe-palm-detection:128x128",
        "mediapipe-palm-detection:192x192",
    ],
    "ImageOutputParser": [
        "esrgan:256x256",
        "rt-super-resolution:50x50",
        "zero-dce:400x600",
        "dncnn3:321x481",
    ],
    "SuperAnimalParser": ["superanimal-landmarker:256x256"],
}
