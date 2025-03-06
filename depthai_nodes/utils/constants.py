import depthai as dai

OUTLINE_COLOR = dai.Color(0.0, 1.0, 0.0, 1.0)
TEXT_COLOR = dai.Color(1, 1, 1, 1)
TEXT_SIZE_PER_HEIGHT = 1 / 30
SMALLER_TEXT_SIZE_PER_HEIGHT = TEXT_SIZE_PER_HEIGHT / 2
TEXT_BACKGROUND_COLOR = dai.Color(0.0, 0.0, 0.0, 0.0)
KEYPOINT_COLOR = dai.Color(1.0, 0.35, 0.367, 1.0)
KEYPOINT_THICKNESS_PER_RESOLUTION = 1 / 300
SMALLER_KEYPOINT_THICKNESS_PER_RESOLUTION = KEYPOINT_THICKNESS_PER_RESOLUTION / 2
DETECTION_FILL_COLOR = dai.Color(21 / 255, 127 / 255, 88 / 255, 0.2)
DETECTION_CORNER_COLOR = dai.Color(21 / 255, 127 / 255, 88 / 255, 1)
DETECTION_CORNER_SIZE = 0.04
DETECTION_BORDER_THICKNESS_PER_RESOLUTION = 1 / 300
SMALLER_DETECTION_THRESHOLD = DETECTION_CORNER_SIZE * 3
SMALLER_DETECTION_CORNER_SIZE = DETECTION_CORNER_SIZE / 2
SMALLER_DETECTION_BORDER_THICKNESS_PER_RESOLUTION = (
    DETECTION_BORDER_THICKNESS_PER_RESOLUTION / 2
)
