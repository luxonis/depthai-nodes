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
    if not isinstance(corners, np.ndarray):
        raise ValueError("Corners must be a numpy array.")
    if corners.shape != (4, 2):
        raise ValueError("Corners must be of shape (4, 2).")

    if corners[0][0] >= corners[1][0]:
        raise ValueError(
            "Corners must be ordered by top-left, top-right, bottom-right, bottom-left. Got top-right corner before top-left."
        )
    if corners[0][1] >= corners[3][1]:
        raise ValueError(
            "Corners must be ordered by top-left, top-right, bottom-right, bottom-left. Got bottom-left corner before top-left."
        )
    if corners[1][1] >= corners[2][1]:
        raise ValueError(
            "Corners must be ordered by top-left, top-right, bottom-right, bottom-left. Got bottom-right corner before top-right."
        )
    if corners[3][0] >= corners[2][0]:
        raise ValueError(
            "Corners must be ordered by top-left, top-right, bottom-right, bottom-left. Got bottom-left corner before bottom-right."
        )

    x_dist = corners[1][0] - corners[0][0]
    y_dist = corners[1][1] - corners[0][1]

    angle = np.degrees(np.arctan2(y_dist, x_dist))
    x_center, y_center = np.mean(corners, axis=0)
    width = np.linalg.norm(corners[0] - corners[1])
    height = np.linalg.norm(corners[1] - corners[2])

    return np.array([x_center, y_center, width, height, angle])


def rotated_bbox_to_corners(
    cx: float, cy: float, w: float, h: float, rotation: float
) -> np.ndarray:
    """Converts a rotated bounding box to the corners of the bounding box.

    @param cx: The x-coordinate of the center of the bounding box.
    @type cx: float
    @param cy: The y-coordinate of the center of the bounding box.
    @type cy: float
    @param w: The width of the bounding box.
    @type w: float
    @param h: The height of the bounding box.
    @type h: float
    @param rotation: The angle of the bounding box given in degrees.
    @type rotation: float
    @return: The corners of the bounding box.
    @rtype: np.ndarray
    """
    if not all(isinstance(i, float) for i in [cx, cy, w, h, rotation]):
        raise ValueError("All inputs must be floats.")
    if not -360 <= rotation <= 360:
        raise ValueError("Rotation must be between -360 and 360.")

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

    return np.array([[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]])


def xyxy_to_xywh(bboxes: np.ndarray) -> np.ndarray:
    """Converts bounding boxes from [x_min, y_min, x_max, y_max] to [x_center, y_center,
    width, height].

    @param bboxes: The bounding boxes to convert.
    @type bboxes: np.ndarray
    @return: The converted bounding boxes.
    @rtype: np.ndarray
    """
    if not isinstance(bboxes, np.ndarray):
        raise ValueError("Bounding boxes must be a numpy array.")
    if len(bboxes.shape) != 2:
        raise ValueError("Bounding boxes must be of shape (N, 4).")

    if bboxes.shape[1] != 4:
        raise ValueError("Bounding boxes must be of shape (N, 4).")

    if not np.all(bboxes[:, :2] <= bboxes[:, 2:]):
        raise ValueError(
            "Bounding box coordinates must be in the format [x_min, y_min, x_max, y_max]."
        )

    x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2
    y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2
    width = bboxes[:, 2] - bboxes[:, 0]
    height = bboxes[:, 3] - bboxes[:, 1]

    return np.stack([x_center, y_center, width, height], axis=-1)


def xywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Convert bounding box coordinates from (x_center, y_center, width, height) to
    (x_min, y_min, x_max, y_max).

    @param bboxes: A numpy array of shape (N, 4) containing the bounding boxes in (x, y, width, height) format.
    @type np.ndarray
    @return: A numpy array of shape (N, 4) containing the bounding boxes in (x_min, y_min, x_max, y_max) format.
    @type np.ndarray
    """
    if not isinstance(bboxes, np.ndarray):
        raise ValueError("Bounding boxes must be a numpy array.")
    if len(bboxes.shape) != 2:
        raise ValueError(
            "Bounding boxes must be of shape (N, 4). Got shape {bboxes.shape}."
        )
    if bboxes.shape[1] != 4:
        raise ValueError(
            "Bounding boxes must be of shape (N, 4). Got shape {bboxes.shape}."
        )

    xyxy_bboxes = np.zeros_like(bboxes)
    xyxy_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x_min = x - w/2
    xyxy_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y_min = y - h/2
    xyxy_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # x_max = x + w/2
    xyxy_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y_max = y + h/2

    return xyxy_bboxes


def top_left_wh_to_xywh(bboxes: np.ndarray) -> np.ndarray:
    """Converts bounding boxes from [top_left_x, top_left_y, width, height] to
    [x_center, y_center, width, height].

    @param bboxes: The bounding boxes to convert.
    @type bboxes: np.ndarray
    @return: The converted bounding boxes.
    @rtype: np.ndarray
    """
    if not isinstance(bboxes, np.ndarray):
        raise ValueError("Bounding boxes must be a numpy array.")
    if len(bboxes.shape) != 2:
        raise ValueError(
            "Bounding boxes must be of shape (N, 4). Got shape {bboxes.shape}."
        )
    if bboxes.shape[1] != 4:
        raise ValueError(
            "Bounding boxes must be of shape (N, 4). Got shape {bboxes.shape}."
        )

    x_center = bboxes[:, 0] + bboxes[:, 2] / 2
    y_center = bboxes[:, 1] + bboxes[:, 3] / 2
    width = bboxes[:, 2]
    height = bboxes[:, 3]

    return np.stack([x_center, y_center, width, height], axis=-1)


def normalize_bboxes(bboxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize bounding box coordinates to (0, 1).

    @param bboxes: A numpy array of shape (N, 4) containing the bounding boxes.
    @type np.ndarray
    @param height: The height of the image.
    @type height: int
    @param width: The width of the image.
    @type width: int
    @return: A numpy array of shape (N, 4) containing the normalized bounding boxes.
    @type np.ndarray
    """
    if not isinstance(bboxes, np.ndarray):
        raise ValueError("Bounding boxes must be a numpy array.")
    if len(bboxes.shape) != 2:
        raise ValueError(
            "Bounding boxes must be of shape (N, 4). Got shape {bboxes.shape}."
        )
    if bboxes.shape[1] != 4:
        raise ValueError(
            "Bounding boxes must be of shape (N, 4). Other options are currently not supported"
        )
    bboxes[:, 0] = bboxes[:, 0] / width
    bboxes[:, 1] = bboxes[:, 1] / height
    bboxes[:, 2] = bboxes[:, 2] / width
    bboxes[:, 3] = bboxes[:, 3] / height

    return bboxes
