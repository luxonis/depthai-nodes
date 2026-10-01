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
    """Decode UFLD lane points from a batched grid tensor.

    Args:
        tensor: Batched logits with grid classes, sampled rows, and lanes as the
            remaining axes; only the first batch item is used.
        row_anchors: Image row positions, in pixels, for the lane sampling grid.
        griding_num: Number of horizontal grid cells, excluding the no-lane class.
        cls_num_per_lane: Number of sampled row positions per lane.
        input_size: Model input size as ``(width, height)``.

    Returns:
        One list per lane containing normalized XY point tuples. Lanes with fewer than
        three valid samples have empty lists.
    """
    return decode_ufld(
        anchors=row_anchors,
        griding_num=griding_num,
        cls_num_per_lane=cls_num_per_lane,
        input_width=input_size[0],
        input_height=input_size[1],
        y=tensor[0],
    )
