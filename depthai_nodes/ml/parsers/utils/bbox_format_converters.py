import numpy as np


def corners_to_rotated_bbox(corners: np.ndarray) -> np.ndarray:
    """Converts the corners of a bounding box to a rotated bounding box.

    @param corners: The corners of the bounding box. The corners are expected to be
        ordered by top-left, top-right, bottom-right, bottom-left.
    @type corners: np.ndarray
    @return: The rotated bounding box defined as [x_center, y_center, width, height,
        angle].
    @rtype: np.ndarray
    """

    x_dist = corners[1][0] - corners[0][0]
    y_dist = corners[1][1] - corners[0][1]

    angle = np.degrees(np.arctan2(y_dist, x_dist))
    x_center, y_center = np.mean(corners, axis=0)
    width = np.linalg.norm(corners[0] - corners[1])
    height = np.linalg.norm(corners[1] - corners[2])

    return np.array([x_center, y_center, width, height, angle])


def rotated_bbox_to_corners(cx, cy, w, h, rotation):
    b = np.cos(rotation) * 0.5
    a = np.sin(rotation) * 0.5
    p0x = cx - a * h - b * w
    p0y = cy + b * h - a * w
    p1x = cx + a * h - b * w
    p1y = cy - b * h - a * w
    p2x = int(2 * cx - p0x)
    p2y = int(2 * cy - p0y)
    p3x = int(2 * cx - p1x)
    p3y = int(2 * cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]]
