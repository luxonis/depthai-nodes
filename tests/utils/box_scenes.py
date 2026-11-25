"""Synthetic box scenes and the label cloud groups ``BoxCloudAssembler`` receives.

A scene is built directly in shared coordinates (mm): each view samples the cuboid faces
it can see, with about 1 mm of noise and zero-depth holes, which are labelled (0, 0, 0)
points. No calibration is involved, because the host stage receives points that are
already transformed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import depthai as dai
import numpy as np

from depthai_nodes.message import BoxDetection, BoxDetections
from depthai_nodes.node.box_associator.box_cloud_assembler import (
    CLOUD_KEY,
    detections_key,
)
from depthai_nodes.node.box_associator.box_label import encode

# Unit normals of the faces a view can see; "+z" is the top.
FACES = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
}
SAMPLE_SPACING_MM = 2.0
NOISE_MM = 1.0
HOLE_FRACTION = 0.1


@dataclass(frozen=True)
class Cuboid:
    """An axis-aligned cuboid standing in shared coordinates (mm)."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]


def sample_faces(
    cuboid: Cuboid,
    faces: Sequence[str],
    seed: int = 0,
    extent: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Return a view's labelled points of the cuboid's faces, holes as zeros.

    Args:
        cuboid: The box.
        faces: Names from ``FACES`` that the view sees.
        seed: Seed of the noise and holes.
        extent: Fraction of each face, along its first in-plane axis, that the
            view sees; the default is the whole face.
    """
    rng = np.random.default_rng(seed)
    center = np.asarray(cuboid.center, dtype=np.float64)
    half = np.asarray(cuboid.size, dtype=np.float64) / 2
    patches = []
    for face in faces:
        normal = FACES[face]
        axis = int(np.flatnonzero(normal)[0])
        u_axis, v_axis = (a for a in range(3) if a != axis)
        u_lo = -half[u_axis] + extent[0] * 2 * half[u_axis]
        u_hi = -half[u_axis] + extent[1] * 2 * half[u_axis]
        u = np.arange(u_lo, u_hi, SAMPLE_SPACING_MM)
        v = np.arange(-half[v_axis], half[v_axis], SAMPLE_SPACING_MM)
        grid_u, grid_v = np.meshgrid(u, v)
        patch = np.zeros((grid_u.size, 3))
        patch[:, axis] = normal[axis] * half[axis]
        patch[:, u_axis] = grid_u.ravel()
        patch[:, v_axis] = grid_v.ravel()
        patches.append(patch + center)
    points = np.concatenate(patches)
    points += rng.normal(0.0, NOISE_MM, points.shape)
    points[rng.random(len(points)) < HOLE_FRACTION] = 0.0
    return points.astype(np.float32)


SceneView = dict[int, np.ndarray]
"""One view's boxes: labelled points by track ID."""


@dataclass(frozen=True)
class Scene:
    """Views of a scene and how many boxes association should find in it."""

    views: list[SceneView]
    boxes: int


def opposite_sides() -> Scene:
    """One box seen from opposite sides, with an overlapping top."""
    box = Cuboid((0.0, 0.0, 100.0), (200.0, 150.0, 200.0))
    return Scene(
        views=[
            {1: sample_faces(box, ["-x", "+z"], seed=0)},
            {1: sample_faces(box, ["+x", "+z"], seed=1)},
        ],
        boxes=1,
    )


def two_separate_boxes() -> Scene:
    """Two boxes far apart, each seen by both views."""
    first = Cuboid((-300.0, 0.0, 75.0), (150.0, 150.0, 150.0))
    second = Cuboid((300.0, 0.0, 75.0), (150.0, 150.0, 150.0))
    return Scene(
        views=[
            {
                1: sample_faces(first, ["-x", "+z"], seed=0),
                2: sample_faces(second, ["-x", "+z"], seed=1),
            },
            {
                1: sample_faces(first, ["+x", "+z"], seed=2),
                2: sample_faces(second, ["+x", "+z"], seed=3),
            },
        ],
        boxes=2,
    )


def two_touching_boxes() -> Scene:
    """Two boxes side by side whose tops touch: a false merge, the accepted cost.

    View 0 tells them apart, while view 1 sees both tops as one box view.
    """
    first = Cuboid((-75.0, 0.0, 75.0), (150.0, 150.0, 150.0))
    second = Cuboid((75.0, 0.0, 75.0), (150.0, 150.0, 150.0))
    return Scene(
        views=[
            {
                1: sample_faces(first, ["-y", "+z"], seed=0),
                2: sample_faces(second, ["-y", "+z"], seed=1),
            },
            {
                1: np.concatenate(
                    [
                        sample_faces(first, ["+y", "+z"], seed=2),
                        sample_faces(second, ["+y", "+z"], seed=3),
                    ]
                )
            },
        ],
        boxes=1,
    )


def partial_view() -> Scene:
    """One box that view 0 sees whole and view 1 sees only part of."""
    box = Cuboid((0.0, 0.0, 100.0), (200.0, 150.0, 200.0))
    return Scene(
        views=[
            {1: sample_faces(box, ["-x", "+z"], seed=0)},
            {1: sample_faces(box, ["+z"], seed=1, extent=(0.0, 0.4))},
        ],
        boxes=1,
    )


def different_track_ids() -> Scene:
    """The same box in both views, under different track IDs."""
    box = Cuboid((0.0, 0.0, 100.0), (200.0, 150.0, 200.0))
    return Scene(
        views=[
            {3: sample_faces(box, ["-x", "+z"], seed=0)},
            {8: sample_faces(box, ["+x", "+z"], seed=1)},
        ],
        boxes=1,
    )


@dataclass(frozen=True)
class ViewPixels:
    """One view's label image and the point each of its pixels lands on."""

    labels: np.ndarray
    """``(H, W, 3)`` ``uint8`` RGB label image."""
    points: np.ndarray
    """``(H, W, 3)`` ``float32`` points in millimetres."""


def paint_view(
    view: int,
    boxes: Mapping[int, np.ndarray],
    background: np.ndarray | None = None,
) -> ViewPixels:
    """Paint a view's boxes as the labeler would, one pixel per point.

    Args:
        view: Index of the view, painted as view + 1 in blue.
        boxes: Labelled points by track ID.
        background: Unlabelled points of the view.
    """
    labels, points = [], []
    for track_id, box_points in boxes.items():
        label = np.array(encode(view, track_id), np.uint8)
        labels.append(np.tile(label, (len(box_points), 1)))
        points.append(box_points)
    if background is not None:
        labels.append(np.zeros((len(background), 3), np.uint8))
        points.append(background)
    flat_labels = np.concatenate(labels or [np.zeros((0, 3), np.uint8)])
    flat_points = np.concatenate(points or [np.zeros((0, 3))]).astype(np.float32)
    return ViewPixels(flat_labels[None], flat_points[None])


def label_cloud(views: Sequence[ViewPixels]) -> dai.PointCloudData:
    """Stand in for the label ``PointCloud``: its organized output with colors.

    Views are concatenated in the given order.
    """
    labels = np.concatenate([v.labels.reshape(-1, 3) for v in views])
    points = np.concatenate([v.points.reshape(-1, 3) for v in views])
    alpha = np.full((len(labels), 1), 255, np.uint8)
    cloud = dai.PointCloudData()
    cloud.setPointsRGB(points, np.hstack([labels, alpha]))
    cloud.setWidth(len(points))
    cloud.setHeight(1)
    return cloud


def box_detection(
    track_id: int,
    bounding_box: dai.RotatedRect | None = None,
    confidence: float = 0.9,
) -> BoxDetection:
    """Return a tracked box detection."""
    if bounding_box is None:
        bounding_box = dai.RotatedRect(
            dai.Rect(dai.Point2f(0.1, 0.1), dai.Point2f(0.4, 0.5))
        )
    return BoxDetection(
        track_id=track_id,
        bounding_box=bounding_box,
        confidence=confidence,
        track_status=dai.Tracklet.TrackingStatus.TRACKED,
    )


def box_detections(items: Sequence[BoxDetection | int]) -> BoxDetections:
    """Return ``BoxDetections`` of the items; an ``int`` is a track ID."""
    detections = [
        box_detection(item) if isinstance(item, int) else item for item in items
    ]
    return BoxDetections(detections, np.full((1, 1), 255, np.uint8))


def box_group(
    cloud: dai.PointCloudData, detections: Sequence[BoxDetections]
) -> dai.MessageGroup:
    """Return the group the pairing ``Sync`` emits: the cloud and each view's
    detections."""
    group = dai.MessageGroup()
    group[CLOUD_KEY] = cloud
    for view, view_detections in enumerate(detections):
        group[detections_key(view)] = view_detections
    return group
