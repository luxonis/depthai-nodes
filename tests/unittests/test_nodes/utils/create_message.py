import depthai as dai
import numpy as np

from depthai_nodes import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Map2D,
    SegmentationMask,
)

HEIGHT, WIDTH = 5, 5
MAX_VALUE = 255
ARR_2D = np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH), dtype=np.int16)
ARR_2D_FLOAT = ARR_2D.astype(np.float32)
IMG = np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH, 3), dtype=np.uint8)
DETS = [
    {"bbox": [0.00, 0.20, 0.00, 0.20], "label": 0, "confidence": 0.2},
    {"bbox": [0.20, 0.40, 0.20, 0.40], "label": 1, "confidence": 0.4},
    {"bbox": [0.40, 0.60, 0.40, 0.60], "label": 2, "confidence": 0.6},
    {"bbox": [0.60, 0.80, 0.60, 0.80], "label": 3, "confidence": 0.8},
    {"bbox": [0.80, 1.00, 0.80, 1.00], "label": 4, "confidence": 1.0},
]


def create_img_frame(img: np.ndarray = IMG, dtype=dai.ImgFrame.Type.BGR888p):
    """Creates a dai.ImgFrame object.

    @param img: Image as a numpy array.
    @type img: np.ndarray
    @return: The created dai.ImgFrame object.
    @rtype: dai.ImgFrame
    """
    assert isinstance(img, np.ndarray)
    img_frame = dai.ImgFrame()
    img_frame.setCvFrame(img, dtype)
    return img_frame


def create_img_detection(det: dict = DETS[0]):
    """Creates a dai.ImgDetection object.

    @param det: Detection dict with keys "bbox" ([xmin, ymin, xmax, ymax]), "label"
        (int), and "confidence" (float).
    @type det: dict
    @return: The created dai.ImgDetection object.
    @rtype: dai.ImgDetection
    """
    assert isinstance(det, dict)
    img_det = dai.ImgDetection()
    img_det.xmin, img_det.ymin, img_det.xmax, img_det.ymax = det["bbox"]
    img_det.label = det["label"]
    img_det.confidence = det["confidence"]
    return img_det


def create_img_detections(dets: list[dict] = DETS):
    """Creates a dai.ImgDetections object.

    @param dets: List of detection dicts, each containing "bbox" ([xmin, ymin, xmax,
        ymax]), "label" (int), and "confidence" (float).
    @type dets: list[dict]
    @return: The created dai.ImgDetections object.
    @rtype: dai.ImgDetections
    """
    assert isinstance(dets, list)
    img_dets = dai.ImgDetections()
    img_dets.detections = [create_img_detection(det) for det in dets]
    return img_dets


def create_img_detection_extended(det: dict = DETS[0]):
    """Creates a ImgDetectionExtended object.

    @param det: Detection dict with keys "bbox" ([xmin, ymin, xmax, ymax]), "label"
        (int), and "confidence" (float).
    @type det: dict
    @return: The created ImgDetectionExtended object.
    @rtype: ImgDetectionExtended
    """
    assert isinstance(det, dict)
    img_det_ext = ImgDetectionExtended()
    xmin, ymin, xmax, ymax = det["bbox"]
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    img_det_ext.label = det["label"]
    img_det_ext.confidence = det["confidence"]
    img_det_ext.rotated_rect = (x_center, y_center, width, height, 0)
    return img_det_ext


def create_img_detections_extended(dets: list[dict] = DETS, mask: np.ndarray = ARR_2D):
    """Creates a ImgDetectionsExtended object.

    @param dets: List of detection dicts, each containing "bbox" ([xmin, ymin, xmax,
        ymax]), "label" (int), and "confidence" (float).
    @type dets: list[dict]
    @return: The created ImgDetectionsExtended object.
    @rtype: ImgDetectionsExtended
    """

    img_dets_ext = ImgDetectionsExtended()
    if dets is not None:
        assert isinstance(dets, list)
        img_dets_ext.detections = [create_img_detection_extended(det) for det in dets]
    if mask is not None:
        img_dets_ext.masks = create_segmentation_mask(mask)
    return img_dets_ext


def create_segmentation_mask(mask: np.ndarray = ARR_2D):
    """Creates a SegmentationMask object.

    @param mask: Segmentation mask as a numpy array.
    @type mask: np.ndarray
    @return: The created SegmentationMask object.
    @rtype: SegmentationMask
    """
    seg_mask = SegmentationMask()
    seg_mask.mask = mask
    return seg_mask


def create_map2d(map: np.ndarray = ARR_2D_FLOAT):
    """Creates a Map2D object.

    @param map: 2D map as a numpy array.
    @type map: np.ndarray
    @return: The created Map2D object.
    @rtype: Map2D
    """
    map2d = Map2D()
    map2d.map = map
    return map2d
