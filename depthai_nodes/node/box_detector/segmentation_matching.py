"""Pair tracked boxes with the detections whose instance masks they own."""

import depthai as dai

Corners = tuple[float, float, float, float]


def match_tracklets_to_detections(
    tracklets: list[dai.Tracklet], detections: list[dai.ImgDetection]
) -> dict[int, int]:
    """Greedily pair tracklets and detections by descending IoU.

    Args:
        tracklets: Tracklets with normalized ROIs.
        detections: Detections with normalized corners.

    Returns:
        Mapping of tracklet index to detection index. Pairs need a positive IoU,
        and each tracklet and detection appears at most once.
    """
    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracklets):
        top_left, bottom_right = track.roi.topLeft(), track.roi.bottomRight()
        track_box = (top_left.x, top_left.y, bottom_right.x, bottom_right.y)
        for det_index, det in enumerate(detections):
            iou = intersection_over_union(
                track_box, (det.xmin, det.ymin, det.xmax, det.ymax)
            )
            if iou > 0.0:
                candidates.append((iou, track_index, det_index))
    candidates.sort(reverse=True)

    matches: dict[int, int] = {}
    used_detections: set[int] = set()
    for _iou, track_index, det_index in candidates:
        if track_index in matches or det_index in used_detections:
            continue
        matches[track_index] = det_index
        used_detections.add(det_index)
    return matches


def intersection_over_union(a: Corners, b: Corners) -> float:
    """Return the IoU of two ``(xmin, ymin, xmax, ymax)`` boxes.

    Args:
        a: First box.
        b: Second box.

    Returns:
        IoU in ``[0, 1]``; zero when the boxes do not overlap.
    """
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0
