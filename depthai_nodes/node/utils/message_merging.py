from typing import List

import depthai as dai
import numpy as np

from depthai_nodes.message.clusters import Clusters
from depthai_nodes.message.img_detections import (
    ImgDetectionsExtended,
)
from depthai_nodes.message.keypoints import Keypoints
from depthai_nodes.message.lines import Lines
from depthai_nodes.message.map import Map2D
from depthai_nodes.message.prediction import Predictions
from depthai_nodes.message.segmentation import SegmentationMask
from depthai_nodes.node.utils.util_constants import UNASSIGNED_MASK_LABEL, GMessage


def merge_messages(
    messages: List[GMessage],
) -> GMessage:
    if len(messages) == 0:
        raise ValueError("Not enough messages to merge")
    if len({type(x) for x in messages}) != 1:
        raise TypeError("Cannot merge messages of different types")
    if all(isinstance(message, dai.ImgDetections) for message in messages):
        return merge_img_detections(messages)  # type: ignore
    elif all(isinstance(message, ImgDetectionsExtended) for message in messages):
        return merge_img_detections_extended(messages)  # type: ignore
    elif all(isinstance(message, SegmentationMask) for message in messages):
        return merge_segmentation_masks(messages)  # type: ignore
    elif all(isinstance(message, Keypoints) for message in messages):
        return merge_keypoints(messages)  # type: ignore
    elif all(isinstance(message, Clusters) for message in messages):
        return merge_clusters(messages)  # type: ignore
    elif all(isinstance(message, Map2D) for message in messages):
        return merge_map2d(messages)  # type: ignore
    elif all(isinstance(message, Lines) for message in messages):
        return merge_lines(messages)  # type: ignore
    elif all(isinstance(message, Predictions) for message in messages):
        return merge_predictions(messages)  # type: ignore
    else:
        raise TypeError(f"Unsupported message types: {type(messages[0])}")


def merge_img_detections(
    detections: List[dai.ImgDetections],
) -> dai.ImgDetections:
    det_list = []
    for det in detections:
        det_list.extend(det.detections)
    new_img_detections = dai.ImgDetections()
    new_img_detections.detections = det_list
    return new_img_detections


def merge_img_detections_extended(detections: List[ImgDetectionsExtended]):
    new_detections_list = []
    for det in detections:
        new_detections_list.extend(det.detections)
    seg_maps = [x.masks for x in detections if x.masks is not None and x.masks.size > 0]
    if len(seg_maps) > 0:
        new_masks = merge_segmentation_mask_array(seg_maps)
    else:
        new_masks = np.empty(0, dtype=np.int16)
    new_detections = ImgDetectionsExtended()
    new_detections.detections = new_detections_list
    new_detections.masks = new_masks
    return new_detections


def merge_segmentation_mask_array(
    segmentation_masks: List[np.ndarray],
) -> np.ndarray:
    if len({x.shape for x in segmentation_masks}) != 1:
        raise ValueError("Segmentation masks must have the same shape")
    full_seg_map = np.full_like(segmentation_masks[0], UNASSIGNED_MASK_LABEL)
    for seg_map in segmentation_masks:
        full_seg_map[seg_map != UNASSIGNED_MASK_LABEL] = seg_map[
            seg_map != UNASSIGNED_MASK_LABEL
        ]
    return full_seg_map


def merge_segmentation_masks(
    segmentation_masks: List[SegmentationMask],
) -> SegmentationMask:
    new_mask = SegmentationMask()
    seg_arrs = [x.mask for x in segmentation_masks]
    new_mask.mask = merge_segmentation_mask_array(seg_arrs)
    return new_mask


def merge_keypoints(
    keypoints: List[Keypoints],
) -> Keypoints:
    new_kpts_list = []
    edges_list = []
    for kpts in keypoints:
        new_edges = [
            (p1 + len(new_kpts_list), p2 + len(new_kpts_list)) for p1, p2 in kpts.edges
        ]
        edges_list.extend(new_edges)
        new_kpts_list.extend(kpts.keypoints)
    new_kpts = Keypoints()
    new_kpts.keypoints = new_kpts_list
    new_kpts.edges = edges_list
    return new_kpts


def merge_clusters(clusters: List[Clusters]) -> Clusters:
    new_clusters_list = []
    for c in clusters:
        new_clusters_list.extend(c.clusters)
    new_clusters = Clusters()
    new_clusters.clusters = new_clusters_list
    return new_clusters


def merge_map2d(maps: List[Map2D]) -> Map2D:
    new_map2d = Map2D()
    map_arrs = [x.map for x in maps]
    new_map2d.map = merge_segmentation_mask_array(map_arrs)
    return new_map2d


def merge_lines(
    lines: List[Lines],
) -> Lines:
    new_lines_list = []
    for line in lines:
        new_lines_list.extend(line.lines)
    new_lines = Lines()
    new_lines.lines = new_lines_list
    return new_lines


def merge_predictions(predictions: List[Predictions]) -> Predictions:
    new_predictions_list = []
    for p in predictions:
        new_predictions_list.extend(p.predictions)
    new_predictions = Predictions()
    new_predictions.predictions = new_predictions_list
    return new_predictions
