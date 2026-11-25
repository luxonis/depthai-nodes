"""Turn a label cloud and each view's detections into one frame's box clouds."""


import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple

import depthai as dai
import numpy as np

from depthai_nodes.message import BoxCloud, BoxClouds, BoxDetections, BoxViewDetection
from depthai_nodes.node.box_associator.box_label import (
    LABEL_MODULUS,
    decode,
    track_label,
)
from depthai_nodes.node.box_associator.host_messages import send_all, stamp_like
from depthai_nodes.node.box_dimensioner.box_geometry import BoxGeometry

logger = logging.getLogger(__name__)

CLOUD_KEY = "cloud"
"""Group key of the organized label ``PointCloudData``."""

# Box views of different views link when at least MIN_LINK_FRACTION of the smaller
# one's points lie within LINK_DISTANCE_MM of the other.
LINK_DISTANCE_MM = 10.0
MIN_LINK_FRACTION = 0.3
# A box seen by several views keeps one point per voxel of this size.
MERGE_VOXEL_SIZE_MM = 3.0
# A key absent for this many consecutive frames is forgotten.
MAX_ABSENT_FRAMES = 30


class Key(NamedTuple):
    """A box view's view and track ID."""

    view: int
    track_id: int


def detections_key(view: int) -> str:
    """Return the group key of a view's ``BoxDetections``."""
    return f"detections/{view}"


@dataclass(frozen=True)
class _BoxView:
    """One view's cleaned points of one box, with its detection."""

    detection: BoxViewDetection
    points: np.ndarray
    depth_coverage: float
    min_corner: np.ndarray
    """The smallest coordinate of its points on each axis."""
    max_corner: np.ndarray
    """The largest coordinate of its points on each axis."""

    @property
    def key(self) -> Key:
        """The box view's (view, track ID) key."""
        return Key(self.detection.view, self.detection.track_id)


class BoxCloudAssembler(dai.node.HostNode):
    """Emit one ``BoxClouds`` per paired label cloud and detections group.

    Attributes:
        out: Output stream of ``BoxClouds``.
    """

    def __init__(self) -> None:
        """Initialize the stage with no known boxes."""
        super().__init__()
        self._geometry = BoxGeometry()
        self._identities = _BoxIdentities()

    def build(self, group: dai.Node.Output) -> "BoxCloudAssembler":
        """Link the pairing ``Sync``'s output.

        Args:
            group: Output producing ``dai.MessageGroup``s keyed by ``CLOUD_KEY`` and
                ``detections_key``.

        Returns:
            This node.
        """
        self.link_args(group)
        return self

    def process(self, group: dai.Buffer) -> None:
        """Assemble and send the box clouds of one frame.

        Args:
            group: The label cloud and each view's detections of one frame.

        Raises:
            TypeError: If ``group`` is not a ``dai.MessageGroup`` holding a
                ``dai.PointCloudData`` and ``BoxDetections``.
        """
        if not isinstance(group, dai.MessageGroup):
            raise TypeError("Expected dai.MessageGroup")
        cloud = group[CLOUD_KEY]
        if not isinstance(cloud, dai.PointCloudData):
            raise TypeError("Expected dai.PointCloudData")
        groups = self._associate(self._box_views(cloud, group))
        box_ids = self._identities.assign(
            [
                _Group(
                    keys=tuple(member.key for member in members),
                    points=sum(len(member.points) for member in members),
                )
                for members in groups
            ]
        )
        message = BoxClouds(
            [
                self._box_cloud(box_id, members)
                for box_id, members in zip(box_ids, groups, strict=True)
            ]
        )
        stamp_like(message, cloud)
        send_all((self.out, message))

    def _box_views(
        self, cloud: dai.PointCloudData, group: dai.MessageGroup
    ) -> list[_BoxView]:
        """Split the cloud by label, pair each part with its detection, clean it."""
        points, colors = cloud.getPointsRGB()
        views, labels = decode(colors)
        labelled = views >= 0
        if not labelled.any():
            return []
        # One integer per (view, label), ordered like the pair: unique over a 1-D
        # array is far cheaper than over rows.
        codes = views[labelled] * LABEL_MODULUS + labels[labelled]
        order = np.argsort(codes, kind="stable")
        uniques, counts = np.unique(codes[order], return_counts=True)
        parts = np.split(points[labelled][order], np.cumsum(counts)[:-1])
        box_views = []
        for code, raw in zip(uniques.tolist(), parts, strict=True):
            view, label = divmod(code, LABEL_MODULUS)
            detection = self._detection(group, view, label)
            if detection is None:
                logger.warning(
                    "Dropping a box view: view %d has no matching detection for "
                    "label %d",
                    view,
                    label,
                )
                continue
            valid = raw[np.any(raw != 0, axis=1)]
            if len(valid) < BoxGeometry.MIN_POINTS:
                continue
            cleaned = self._geometry.clean(valid)
            if len(cleaned) < BoxGeometry.MIN_POINTS:
                continue
            box_points = cleaned.astype(np.float32)
            box_views.append(
                _BoxView(
                    detection=detection,
                    points=box_points,
                    depth_coverage=len(valid) / len(raw),
                    # float64, so a bounds gap is exact and never exceeds a distance.
                    min_corner=box_points.min(axis=0).astype(np.float64),
                    max_corner=box_points.max(axis=0).astype(np.float64),
                )
            )
        return box_views

    @staticmethod
    def _detection(
        group: dai.MessageGroup, view: int, label: int
    ) -> BoxViewDetection | None:
        """Return the view's first detection whose track ID matches the label."""
        detections = group[detections_key(view)]
        if not isinstance(detections, BoxDetections):
            raise TypeError("Expected BoxDetections")
        for item in detections.items:
            if track_label(item.track_id) == label:
                return BoxViewDetection(
                    view=view,
                    track_id=item.track_id,
                    bounding_box=item.bounding_box,
                    confidence=item.confidence,
                )
        return None

    @staticmethod
    def _associate(box_views: list[_BoxView]) -> list[list[_BoxView]]:
        """Group the box views that show the same box, in a fixed order.

        Groups are the connected components of the links between box views of different
        views.
        """
        parents = list(range(len(box_views)))

        def root(index: int) -> int:
            while parents[index] != index:
                index = parents[index]
            return index

        for first, second in combinations(range(len(box_views)), 2):
            a, b = box_views[first], box_views[second]
            if (
                a.detection.view != b.detection.view
                and not _apart(a, b)
                and _linked(a.points, b.points)
            ):
                parents[root(first)] = root(second)
        groups: dict[int, list[_BoxView]] = {}
        for index, box_view in enumerate(box_views):
            groups.setdefault(root(index), []).append(box_view)
        return list(groups.values())

    @staticmethod
    def _box_cloud(box_id: int, members: list[_BoxView]) -> BoxCloud:
        """Return the box cloud of one association group."""
        return BoxCloud(
            box_id=box_id,
            points=_merge([member.points for member in members]),
            depth_coverage=max(member.depth_coverage for member in members),
            detections=tuple(member.detection for member in members),
        )


def _apart(first: _BoxView, second: _BoxView) -> bool:
    """Return whether the box views' bounds are over ``LINK_DISTANCE_MM`` apart.

    No point of either can then lie within ``LINK_DISTANCE_MM`` of the other, so they
    can't link.
    """
    gaps = np.maximum(
        first.min_corner - second.max_corner, second.min_corner - first.max_corner
    )
    return bool(np.any(gaps > LINK_DISTANCE_MM))


def _linked(first: np.ndarray, second: np.ndarray) -> bool:
    """Return whether enough of the smaller box view lies near the other."""
    # open3d is an optional extra, so import it only when associating.
    import open3d as o3d

    smaller, other = sorted((first, second), key=len)
    clouds = [
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
        for points in (smaller, other)
    ]
    distances = np.asarray(clouds[0].compute_point_cloud_distance(clouds[1]))
    near = np.count_nonzero(distances <= LINK_DISTANCE_MM)
    return bool(near / len(smaller) >= MIN_LINK_FRACTION)


def _merge(views: list[np.ndarray]) -> np.ndarray:
    """Return one box view's points unchanged, or several thinned to one per voxel."""
    if len(views) == 1:
        return views[0]
    points = np.concatenate(views)
    voxels = np.floor(points / MERGE_VOXEL_SIZE_MM).astype(np.int64)
    voxels -= voxels.min(axis=0)
    # One integer per voxel: unique over a 1-D array is far cheaper than over rows.
    codes = np.ravel_multi_index(voxels.T, tuple(voxels.max(axis=0) + 1))
    _, first = np.unique(codes, return_index=True)
    return points[np.sort(first)]


@dataclass(frozen=True)
class _Group:
    """An association group as box identity sees it."""

    keys: tuple[Key, ...]
    points: int
    """The summed points of its cleaned box views."""


@dataclass
class _Claim:
    """A group's claim on existing ``box_id``s while IDs are handed out."""

    candidates: list[int]
    """The IDs its present keys map to, once per key."""
    points: int
    box_id: int | None = None
    """The ID it took, or ``None`` while it has none."""


class _BoxIdentities:
    """Give each association group a ``box_id`` that lasts across frames."""

    def __init__(self) -> None:
        """Start with no known keys."""
        self._box_ids: dict[Key, int] = {}
        self._absent: dict[Key, int] = {}
        self._next_box_id = 0

    def assign(self, groups: Sequence[_Group]) -> list[int]:
        """Return each group's ``box_id`` and age the keys that are not present.

        Args:
            groups: The frame's groups, in a fixed order.

        Returns:
            One ``box_id`` per group, in the order of ``groups``.
        """
        claims = [
            _Claim(
                candidates=[
                    self._box_ids[key] for key in group.keys if key in self._box_ids
                ],
                points=group.points,
            )
            for group in groups
        ]
        for box_id in sorted(
            {box_id for claim in claims for box_id in claim.candidates}
        ):
            winner = _choose_group(box_id, claims)
            if winner is not None:
                claims[winner].box_id = box_id
        box_ids = []
        for claim in claims:
            box_id = claim.box_id
            if box_id is None:
                box_id = self._next_box_id
                self._next_box_id += 1
            box_ids.append(box_id)
        present = set()
        for group, box_id in zip(groups, box_ids, strict=True):
            for key in group.keys:
                self._box_ids[key] = box_id
                present.add(key)
        self._age(present)
        return box_ids

    def _age(self, present: set[Key]) -> None:
        """Reset present keys and forget keys absent for ``MAX_ABSENT_FRAMES``."""
        for key in list(self._box_ids):
            absent = 0 if key in present else self._absent.get(key, 0) + 1
            if absent >= MAX_ABSENT_FRAMES:
                del self._box_ids[key]
                self._absent.pop(key, None)
            else:
                self._absent[key] = absent


def _choose_group(box_id: int, claims: Sequence[_Claim]) -> int | None:
    """Return the index of the claim without an ID that should take ``box_id``.

    It is the claim with the most keys mapping to the ID, then the most points,
    then the first; ``None`` when no claim without an ID has a key mapping to it.
    """
    best = None
    best_rank = (0, 0)
    for index, claim in enumerate(claims):
        if claim.box_id is not None:
            continue
        rank = (claim.candidates.count(box_id), claim.points)
        if rank[0] > 0 and rank > best_rank:
            best, best_rank = index, rank
    return best
