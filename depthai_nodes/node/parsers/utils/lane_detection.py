import numpy as np

from .ufld import decode_ufld


def compute_lane_detection_points(
    tensor: np.ndarray,
    *,
    row_anchors: list[int],
    griding_num: int,
    cls_num_per_lane: int,
    input_size: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    """Decode lane points from the UFLD output tensor."""
    return decode_ufld(
        anchors=row_anchors,
        griding_num=griding_num,
        cls_num_per_lane=cls_num_per_lane,
        input_width=input_size[0],
        input_height=input_size[1],
        y=tensor[0],
    )
