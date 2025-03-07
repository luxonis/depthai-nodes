#!/bin/bash

# Usage: ./run_e2e_tests.sh <DEPTHAI_VERSION> <FLAGS...>

DEPTHAI_VERSION=$1
rm -rf venv
python3 -m venv venv 
source venv/bin/activate
export LC_ALL=en_US.UTF-8

# Check if the DEPTHAI_VERSION is experimental
if [[ "$DEPTHAI_VERSION" == *"experimental"* ]]; then
    export EXPERIMENTAL_DEPTHAI=true
    export PYTHONPATH=$PYTHONPATH:/tmp/depthai-core/build/bindings/python
    echo "DEPTHAI_VERSION is experimental. Skipping installation and setting EXPERIMENTAL_DEPTHAI=true."
else
    LUXONIS_EXTRA_INDEX_URL=$4
    pip install --extra-index-url "$LUXONIS_EXTRA_INDEX_URL" depthai=="$DEPTHAI_VERSION"
fi

pip install -e .
pip install -r requirements-dev.txt

# Source camera IPs and run main script
cd tests/end_to_end
source <(python3 setup_camera_ips.py)
export HUBAI_TEAM_SLUG=$2
export HUBAI_API_KEY=$3
export DISPLAY=:99
if [[ $5 == "v0.1.2-alpha" ]]; then
    python3 main.py --platform RVC4 -m deeplab-v3-plus:512x288 deeplab-v3-plus:512x512 lite-hrnet:18-coco-192x256 lite-hrnet:18-coco-288x384 lite-hrnet:30-coco-192x256 lite-hrnet:30-coco-288x384 qrdet:nano-512x288 yunet:320x240 yunet:640x480 yunet:960x720 yunet:1280x960 dncnn3:320x240 dncnn3:640x480 zero-dce:600x400 dm-count:qnrf-960x540 dm-count:qnrf-640x360 dm-count:qnrf-426x240 dm-count:shb-960x540 dm-count:shb-640x360 dm-count:shb-256x144 dm-count:sha-640x360 mediapipe-selfie-segmentation:256x144 dm-count:sha-426x240 age-gender-recognition:62x62 ewasr:512x384 pp-liteseg:1024x512 midas-v2-1:small-384x256 midas-v2-1:small-512x288 midas-v2-1:small-256x192 midas-v2-1:small-512x384 midas-v2-1:small-1024x768 depth-anything-v2:vit-s-336x252 depth-anything-v2:vit-s-560x420 depth-anything-v2:vit-s-mde-indoors-336x252 depth-anything-v2:vit-s-mde-outdoors-336x252 thermal-person-detection:256x192 efficientvit:b1-224x224 scrfd-person-detection:25g-640x640 yolo-p:bdd100k-320x320 l2cs-net:448x448 osnet:market1501-128x256 osnet:imagenet-128x256 osnet:multi-source-domain-128x256 arcface:lfw-112x112 mediapipe-hand-landmarker:224x224 head-pose-estimation:60x60 objectron:camera-224x224 objectron:chair-224x224 objectron:cup-224x224 objectron:sneakers-224x224 ppe-detection:640x640 paddle-text-recognition:320x48 paddle-text-detection:256x256 paddle-text-detection:544x960 paddle-text-detection:320x576 ultra-fast-lane-detection:culane-800x288 ultra-fast-lane-detection:tusimple-800x288 deeplab-v3-plus:256x256 deeplab-v3-plus:person-513x513 deeplab-v3-plus:person-256x256 deeplab-v3-plus:513x513 yolov6-nano:coco-416x416 yolov8-large-pose-estimation:coco-640x352 yolov8-nano-pose-estimation:coco-512x288 yolov8-instance-segmentation-nano:coco-512x288 yolov8-instance-segmentation-large:coco-640x352 fastsam-x:640x352 fastsam-s:512x288 yolov10-nano:coco-512x288 m-lsd:512x512 m-lsd-tiny:512x512 emotion-recognition:gray-64x64 image-quality-assessment:256x256 efficientnet-lite:lite4-300x300 efficientnet-lite:lite0-224x224 mobilenet-ssd:300x300 license-plate-detection:640x640 vehicle-attributes-classification:72x72 emotion-recognition:260x260 mediapipe-palm-detection:128x128 esrgan:256x256 rt-super-resolution:50x50 yolov6-large:r2-coco-640x352 yolov6-nano:r2-coco-512x288 superanimal-landmarker:256x256 mediapipe-palm-detection:192x192 mediapipe-face-landmarker:192x192 scrfd-face-detection:10g-640x640
else
    python3 main.py --platform RVC4 --depthai-nodes-version $5
fi