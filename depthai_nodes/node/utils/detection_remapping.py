import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.classification import Classifications
from depthai_nodes.message.clusters import Cluster, Clusters
from depthai_nodes.message.img_detections import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
)
from depthai_nodes.message.keypoints import Keypoint, Keypoints
from depthai_nodes.message.lines import Line, Lines
from depthai_nodes.message.map import Map2D
from depthai_nodes.message.prediction import Prediction, Predictions
from depthai_nodes.message.segmentation import SegmentationMask
from depthai_nodes.node.utils.util_constants import UNASSIGNED_MASK_LABEL, GMessage


def remap_message(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    message: GMessage,
) -> GMessage:
    if isinstance(message, dai.ImgDetections):
        return remap_img_detections(src_transformation, dst_transformation, message)
    elif isinstance(message, Keypoints):
        return remap_keypoints(src_transformation, dst_transformation, message)
    elif isinstance(message, SegmentationMask):
        return remap_segmentation_mask(src_transformation, dst_transformation, message)
    elif isinstance(message, ImgDetectionsExtended):
        return remap_img_detections_extended(
            src_transformation, dst_transformation, message
        )
    elif isinstance(message, Clusters):
        return remap_clusters(src_transformation, dst_transformation, message)
    elif isinstance(message, Map2D):
        return remap_map2d(src_transformation, dst_transformation, message)
    elif isinstance(message, Lines):
        return remap_lines(src_transformation, dst_transformation, message)
    elif isinstance(message, Predictions):
        return remap_predictions(src_transformation, dst_transformation, message)
    elif isinstance(message, Classifications):
        return remap_classifications(src_transformation, dst_transformation, message)
    else:
        raise TypeError(f"Unsupported message type: {type(message)}")


def remap_img_detections(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    detections: dai.ImgDetections,
) -> dai.ImgDetections:
    new_detections = dai.ImgDetections()
    new_detections.detections = [
        remap_img_detection(src_transformation, dst_transformation, det)
        for det in detections.detections
    ]
    return new_detections


def remap_img_detections_extended(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    detections: ImgDetectionsExtended,
) -> ImgDetectionsExtended:
    new_detections = ImgDetectionsExtended()
    new_detections.detections = [
        remap_img_detection_extended(src_transformation, dst_transformation, det)
        for det in detections.detections
    ]
    if detections.masks.size > 0:
        new_detections.masks = remap_segmentation_mask_array(
            src_transformation, dst_transformation, detections.masks
        )
    else:
        new_detections.masks = np.empty(0, dtype=np.int16)
    return new_detections


def remap_segmentation_mask_array(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    segmentation_mask: np.ndarray,
) -> np.ndarray:
    dst_matrix = np.array(dst_transformation.getMatrix())
    src_matrix = np.array(src_transformation.getMatrixInv())
    trans_matrix = dst_matrix @ src_matrix
    new_mask = cv2.warpPerspective(
        segmentation_mask,
        trans_matrix,
        dst_transformation.getSize(),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=UNASSIGNED_MASK_LABEL,  # type: ignore
    )
    return new_mask


def remap_segmentation_mask(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    segmentation_mask: SegmentationMask,
) -> SegmentationMask:
    new_mask = SegmentationMask()
    new_mask.mask = remap_segmentation_mask_array(
        src_transformation, dst_transformation, segmentation_mask.mask
    )
    return new_mask


def remap_img_detection(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    img_detection: dai.ImgDetection,
) -> dai.ImgDetection:
    new_det = dai.ImgDetection()
    min_pt = src_transformation.remapPointTo(
        dst_transformation,
        dai.Point2f(
            np.clip(img_detection.xmin, 0, 1), np.clip(img_detection.ymin, 0, 1)
        ),
    )
    max_pt = src_transformation.remapPointTo(
        dst_transformation,
        dai.Point2f(
            np.clip(img_detection.xmax, 0, 1), np.clip(img_detection.ymax, 0, 1)
        ),
    )
    new_det.xmin = max(0, min(min_pt.x, 1))
    new_det.ymin = max(0, min(min_pt.y, 1))
    new_det.xmax = max(0, min(max_pt.x, 1))
    new_det.ymax = max(0, min(max_pt.y, 1))
    new_det.label = img_detection.label
    new_det.confidence = img_detection.confidence
    return new_det


def remap_img_detection_extended(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    detection: ImgDetectionExtended,
) -> ImgDetectionExtended:
    new_det = ImgDetectionExtended()

    if detection.rotated_rect.angle == 0:
        new_rect = src_transformation.remapRectTo(
            dst_transformation, detection.rotated_rect
        )
        new_det.rotated_rect = (
            new_rect.center.x,
            new_rect.center.y,
            new_rect.size.width,
            new_rect.size.height,
            new_rect.angle,
        )
    else:
        # TODO: This is a temporary fix - DepthAI doesn't handle rotated rects with angle != 0 correctly
        pts = detection.rotated_rect.getPoints()
        pts = [dai.Point2f(np.clip(pt.x, 0, 1), np.clip(pt.y, 0, 1)) for pt in pts]
        remapped_pts = [
            src_transformation.remapPointTo(dst_transformation, pt) for pt in pts
        ]
        remapped_pts = [
            (np.clip(pt.x, 0, 1), np.clip(pt.y, 0, 1)) for pt in remapped_pts
        ]
        (center_x, center_y), (width, height), angle = cv2.minAreaRect(
            np.array(remapped_pts, dtype=np.float32)
        )
        new_det.rotated_rect = (
            center_x,
            center_y,
            width,
            height,
            angle,
        )
    new_det.confidence = detection.confidence
    new_det.label = detection.label
    new_det.label_name = detection.label_name

    new_kpts_list = []
    for kpt in detection.keypoints:
        new_kpt = remap_keypoint(src_transformation, dst_transformation, kpt)
        new_kpts_list.append(new_kpt)
    new_kpts = Keypoints()
    new_kpts.keypoints = new_kpts_list
    new_det.keypoints = new_kpts
    return new_det


def remap_keypoint(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    keypoint: Keypoint,
) -> Keypoint:
    new_kpt = Keypoint()
    new_kpt.x = src_transformation.remapPointTo(
        dst_transformation, dai.Point2f(keypoint.x, keypoint.y)
    ).x
    new_kpt.y = src_transformation.remapPointTo(
        dst_transformation, dai.Point2f(keypoint.x, keypoint.y)
    ).y
    new_kpt.z = keypoint.z
    new_kpt.confidence = keypoint.confidence
    new_kpt.label_name = keypoint.label_name
    return new_kpt


def remap_keypoints(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    keypoints: Keypoints,
) -> Keypoints:
    new_kpts_list = []
    for kpt in keypoints.keypoints:
        new_kpt = remap_keypoint(src_transformation, dst_transformation, kpt)
        new_kpts_list.append(new_kpt)
    new_kpts = Keypoints()
    new_kpts.keypoints = new_kpts_list
    return new_kpts


def remap_clusters(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    clusters: Clusters,
) -> Clusters:
    new_clusters = Clusters()
    new_clusters.clusters = [
        remap_cluster(src_transformation, dst_transformation, cluster)
        for cluster in clusters.clusters
    ]
    return new_clusters


def remap_cluster(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    cluster: Cluster,
) -> Cluster:
    new_cluster = Cluster()
    new_cluster.label = cluster.label
    new_cluster.points = [
        src_transformation.remapPointTo(dst_transformation, pt) for pt in cluster.points
    ]
    return new_cluster


def remap_map2d(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    map2d: Map2D,
) -> Map2D:
    new_map2d = Map2D()
    new_map_arr = remap_segmentation_mask_array(
        src_transformation, dst_transformation, map2d.map
    )
    new_map2d.map = new_map_arr
    return new_map2d


def remap_lines(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    lines: Lines,
) -> Lines:
    new_lines = Lines()
    new_lines.lines = [
        remap_line(src_transformation, dst_transformation, line) for line in lines.lines
    ]
    return new_lines


def remap_line(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    line: Line,
) -> Line:
    new_line = Line()
    new_line.confidence = line.confidence
    new_line.start_point = src_transformation.remapPointTo(
        dst_transformation, line.start_point
    )
    new_line.end_point = src_transformation.remapPointTo(
        dst_transformation, line.end_point
    )
    return new_line


def remap_predictions(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    predictions: Predictions,
) -> Predictions:
    new_predictions = Predictions()
    new_predictions.predictions = [
        remap_prediction(src_transformation, dst_transformation, prediction)
        for prediction in predictions.predictions
    ]
    return new_predictions


def remap_prediction(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    prediction: Prediction,
) -> Prediction:
    new_prediction = Prediction()
    new_prediction.prediction = prediction.prediction
    return new_prediction


def remap_classifications(
    src_transformation: dai.ImgTransformation,
    dst_transformation: dai.ImgTransformation,
    classifications: Classifications,
) -> Classifications:
    new_classifications = Classifications()
    new_classifications.classes = classifications.classes
    new_classifications.scores = classifications.scores
    return new_classifications
