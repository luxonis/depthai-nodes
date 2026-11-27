"""The label colour of a box view, shared by the labeler and the host stage.

A box pixel is (index, 0, view + 1), where the index is the box's position in its
view's ``BoxDetections`` items; background is black.
"""

import numpy as np

INDEX_COUNT = 256
"""The number of indices a label can hold, one byte."""


def encode(view: int, index: int) -> tuple[int, int, int]:
    """Return the RGB colour of the pixels of the view's ``index``-th detection."""
    return index, 0, view + 1


def decode(colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the view and index of each ``(N, 3+)`` colour; background has view -1."""
    channels = colors.astype(np.int64)
    return channels[:, 2] - 1, channels[:, 0]
