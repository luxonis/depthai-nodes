import depthai as dai

from depthai_nodes.node.box_detector.segmentation_matching import (
    match_tracklets_to_detections,
)


def tracklet(box: tuple[float, float, float, float]) -> dai.Tracklet:
    xmin, ymin, xmax, ymax = box
    result = dai.Tracklet()
    result.roi = dai.Rect(dai.Point2f(xmin, ymin), dai.Point2f(xmax, ymax))
    return result


def detection(box: tuple[float, float, float, float]) -> dai.ImgDetection:
    result = dai.ImgDetection()
    result.xmin, result.ymin, result.xmax, result.ymax = box
    return result


def test_each_tracklet_pairs_with_its_overlapping_detection():
    tracklets = [tracklet((0.6, 0.6, 0.9, 0.9)), tracklet((0.1, 0.1, 0.3, 0.3))]
    detections = [detection((0.1, 0.1, 0.3, 0.3)), detection((0.6, 0.6, 0.9, 0.9))]

    assert match_tracklets_to_detections(tracklets, detections) == {0: 1, 1: 0}


def test_highest_overlap_wins_a_contested_detection():
    tracklets = [tracklet((0.1, 0.1, 0.3, 0.4)), tracklet((0.1, 0.1, 0.3, 0.3))]
    detections = [detection((0.1, 0.1, 0.3, 0.3))]

    assert match_tracklets_to_detections(tracklets, detections) == {1: 0}


def test_tracklet_without_overlap_stays_unmatched():
    tracklets = [tracklet((0.0, 0.8, 0.1, 0.9))]
    detections = [detection((0.1, 0.1, 0.3, 0.3))]

    assert match_tracklets_to_detections(tracklets, detections) == {}


def test_touching_boxes_do_not_match():
    tracklets = [tracklet((0.0, 0.0, 0.1, 0.1))]
    detections = [detection((0.1, 0.0, 0.2, 0.1))]

    assert match_tracklets_to_detections(tracklets, detections) == {}
