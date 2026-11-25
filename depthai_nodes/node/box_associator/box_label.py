"""The label colour of a box view, shared by the labeler and the host stage.

A box pixel is (``label & 0xFF``, ``label >> 8``, view + 1), where the label is the
track ID mod ``LABEL_MODULUS``; background is black.
"""

import numpy as np

LABEL_MODULUS = 65536
"""The track ID is painted mod this into the R and G channels."""


def track_label(track_id: int) -> int:
    """Return the label a track ID is painted as."""
    return track_id % LABEL_MODULUS


def encode(view: int, track_id: int) -> tuple[int, int, int]:
    """Return the RGB colour of the track's pixels in the view."""
    label = track_label(track_id)
    return label & 0xFF, label >> 8, view + 1


def decode(colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the view and label of each ``(N, 3+)`` colour; background has view -1."""
    channels = colors.astype(np.int64)
    return channels[:, 2] - 1, channels[:, 0] | channels[:, 1] << 8
