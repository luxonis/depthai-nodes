import numpy as np


def xywh2xyxy(bboxes):
    """ 
    Convert bounding box coordinates from (x, y, width, height) to (x_min, y_min, x_max, y_max).

    @param bboxes: A numpy array of shape (N, 4) containing the bounding boxes in (x, y, width, height) format.
    @type np.ndarray
    @return: A numpy array of shape (N, 4) containing the bounding boxes in (x_min, y_min, x_max, y_max) format.
    @type np.ndarray
    """

    xyxy_bboxes = np.zeros_like(bboxes)
    xyxy_bboxes[:, 0] = bboxes[:, 0] # x_min = x
    xyxy_bboxes[:, 1] = bboxes[:, 1] # y_min = y
    xyxy_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] # x_max = x + w
    xyxy_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] # y_max = y + h
    return xyxy_bboxes

def normalize_bbox(bbox, height, width):
    """
    Normalize bounding box coordinates to (0, 1).

    @param bbox: A tuple or list with 4 elements (x_min, y_min, w, h).
    @type bbox: tuple or list
    @param height: The height of the image.
    @type height: int
    @param width: The width of the image.
    @type width: int
    @return: A list with 4 elements [x_min, y_min, w, h].
    @type list
    """

    xmin, ymin, w, h = bbox
    return [xmin / width, ymin / height, w / width, h / height]

def normalize_bboxes(bboxes, height, width):
    """
    Normalize bounding box coordinates to (0, 1).

    @param bboxes: A numpy array of shape (N, 4) containing the bounding boxes.
    @type np.ndarray
    @param height: The height of the image.
    @type height: int
    @param width: The width of the image.
    @type width: int
    @return: A numpy array of shape (N, 4) containing the normalized bounding boxes.
    @type np.ndarray
    """

    return np.array([normalize_bbox(bbox, height, width) for bbox in bboxes])
