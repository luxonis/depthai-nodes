from math import ceil

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import native
from .scene import Cuboid, PointCloud, Scene, Segment, SupportPlane

_PAIR_QUERY_NEIGHBOR_BUDGET = 8_000_000
_BATCH_QUERY_NEIGHBOR_BUDGET = 2_000_000


def _voxel_centroids(
    points_mm: NDArray,
    size_mm: float,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    accelerated = native.voxel_centroids(points_mm, size_mm)
    if accelerated is not None:
        return accelerated
    grid = np.floor(points_mm / size_mm)
    low = grid.min(axis=0).astype(np.int64)
    span = (grid.max(axis=0) - low + 1).astype(np.int64)
    if int(span[0]) * int(span[1]) * int(span[2]) < 8_000_000:
        integer = grid.astype(np.int64) - low
        keys = (integer[:, 0] * span[1] + integer[:, 1]) * span[2] + integer[:, 2]
        dense_counts = np.bincount(keys)
        occupied = dense_counts != 0
        counts = dense_counts[occupied]
        point_to_voxel = (np.cumsum(occupied) - 1)[keys]
    else:
        _, point_to_voxel, counts = np.unique(
            grid,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
    centroids = np.column_stack(
        [
            np.bincount(point_to_voxel, weights=axis, minlength=len(counts)) / counts
            for axis in points_mm.T
        ]
    )
    return centroids, point_to_voxel


def _components(
    rows: NDArray, columns: NDArray, size: int
) -> tuple[int, NDArray[np.int32]]:
    graph = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, columns)),
        shape=(size, size),
    )
    return connected_components(graph, directed=False)


def _roots(parents: NDArray[np.int32], nodes: NDArray[np.int32]) -> NDArray[np.int32]:
    roots = parents[nodes]
    while np.any(roots != parents[roots]):
        roots = parents[roots]
    parents[nodes] = roots
    return roots


def _cluster_voxels(
    centroids: NDArray[np.float64],
    radius_mm: float,
    size_mm: float,
) -> tuple[int, NDArray]:
    tree = cKDTree(centroids)
    size = len(centroids)
    span = 2 * ceil(radius_mm / size_mm) + 1
    max_neighbors = min(size, span**3)
    if size * max_neighbors <= _PAIR_QUERY_NEIGHBOR_BUDGET:
        rows, columns = tree.query_pairs(radius_mm, output_type="ndarray").T
        return _components(rows, columns, size)

    # Store only one batch of KD-tree edges. Collapse it to a forest of
    # representatives before querying the next batch.
    batch_size = max(1, _BATCH_QUERY_NEIGHBOR_BUDGET // max_neighbors)
    nodes = np.arange(size, dtype=np.int32)
    parents = nodes.copy()
    weights = np.ones(size, dtype=np.int32)
    for start in range(0, size, batch_size):
        batch = nodes[start : start + batch_size]
        neighbors = tree.query_ball_point(
            centroids[batch], radius_mm, return_sorted=False
        )
        rows = np.repeat(batch, list(map(len, neighbors)))
        columns = np.concatenate(neighbors, dtype=np.int32)
        del neighbors
        forward = columns > rows
        left = _roots(parents, rows[forward])
        right = _roots(parents, columns[forward])
        distinct = left != right
        if not distinct.any():
            continue
        representatives, endpoints = np.unique(
            np.concatenate((left[distinct], right[distinct])),
            return_inverse=True,
        )
        count, labels = _components(*np.split(endpoints, 2), len(representatives))
        sizes = weights[representatives]
        largest = np.zeros(count, dtype=np.int32)
        np.maximum.at(largest, labels, sizes)
        leaders = sizes == largest[labels]
        chosen = np.full(count, size, dtype=np.int32)
        np.minimum.at(chosen, labels[leaders], representatives[leaders])
        parents[representatives] = chosen[labels]
        weights[chosen] = np.bincount(labels, weights=sizes, minlength=count).astype(
            np.int32
        )

    roots = _roots(parents, nodes)
    first = np.full(size, size, dtype=np.int32)
    np.minimum.at(first, roots, nodes)
    groups, labels = np.unique(first[roots], return_inverse=True)
    return len(groups), labels


def detect(
    scene: Scene,
    *,
    foreground_min_height_mm: float,
    foreground_max_height_mm: float,
    voxel_size_mm: float,
    cluster_radius_mm: float,
    cluster_min_voxels: int,
    cluster_min_points: int,
) -> list[Segment]:
    xyz = scene.cloud.xyz_mm
    if len(xyz) > np.iinfo(np.int32).max + 1:
        raise ValueError("Cloud has too many rows for int32 segment indices")
    prepared = native.detect_voxels(
        scene, foreground_min_height_mm, foreground_max_height_mm, voxel_size_mm
    )
    if prepared is not None:
        rows, centroids, point_to_voxel = prepared
        if len(rows) < cluster_min_points:
            return []
    else:
        rows = native.foreground_rows(
            xyz, scene.plane, foreground_min_height_mm, foreground_max_height_mm
        )
        if rows is None:
            distances = (
                np.einsum("ij,j->i", xyz, scene.plane.normal) + scene.plane.offset_mm
            )
            rows = np.flatnonzero(
                (distances >= foreground_min_height_mm)
                & (distances <= foreground_max_height_mm)
            ).astype(np.int32)
        if len(rows) < cluster_min_points:
            return []

        accelerated = native.voxel_centroids(xyz, voxel_size_mm, rows)
        centroids, point_to_voxel = (
            accelerated
            if accelerated is not None
            else _voxel_centroids(xyz[rows], voxel_size_mm)
        )
    if len(centroids) < cluster_min_voxels:
        return []
    count, voxel_labels = _cluster_voxels(centroids, cluster_radius_mm, voxel_size_mm)
    point_labels = voxel_labels[point_to_voxel]
    voxels_per_cluster = np.bincount(voxel_labels, minlength=count)
    points_per_cluster = np.bincount(point_labels, minlength=count)
    accepted = (voxels_per_cluster >= cluster_min_voxels) & (
        points_per_cluster >= cluster_min_points
    )
    if not accepted.any():
        return []
    included = accepted[point_labels]
    ordered_rows = native.group_rows(rows, point_labels, accepted, points_per_cluster)
    if ordered_rows is None:
        ordered_rows = rows[included][np.argsort(point_labels[included], kind="stable")]
    starts = np.cumsum(points_per_cluster[accepted])[:-1]
    return [Segment(indices) for indices in np.split(ordered_rows, starts)]


def _supported_extreme(
    values: NDArray,
    band_mm: float,
    min_cells: int,
    *,
    quantile: float | None = None,
    quantile_window_mm: float | None = None,
    preserve_separated_extents: bool = False,
) -> float | None:
    values.sort()
    starts = np.searchsorted(values, values - band_mm)
    counts = np.arange(1, len(values) + 1) - starts
    supported = values[counts >= min_cells]
    if quantile is not None and len(supported) > 1:
        near_edge = values
        near_edge_supported = supported
        if quantile_window_mm is not None:
            cutoff = supported[-1] - quantile_window_mm
            near_edge = values[values >= cutoff]
            near_edge_supported = supported[supported >= cutoff]
        if preserve_separated_extents:
            through_highest_supported = near_edge[near_edge <= supported[-1]]
            gaps = np.flatnonzero(np.diff(through_highest_supported) > band_mm)
            if len(gaps):
                near_edge = through_highest_supported[gaps[-1] + 1 :]
                near_edge_supported = near_edge_supported[
                    near_edge_supported >= near_edge[0]
                ]
        trimmed = near_edge_supported[
            near_edge_supported <= np.quantile(near_edge, quantile)
        ]
        if len(trimmed):
            supported = trimmed
    return float(supported[-1]) if len(supported) else None


def _footprint_bounds(
    xy: NDArray[np.float64],
    cell_mm: float,
    band_mm: float,
    min_cells: int,
    min_side_mm: float,
    *,
    quantile: float | None = None,
    quantile_window_mm: float | None = None,
    preserve_separated_extents: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lows, highs = xy.min(axis=0), xy.max(axis=0)
    for axis in range(2):
        accelerated = native.cross_extrema(xy, axis, cell_mm)
        if accelerated is None:
            _, cross_cells = np.unique(
                np.floor(xy[:, 1 - axis] / cell_mm), return_inverse=True
            )
            count = cross_cells.max() + 1
        for side, (sign, bounds) in enumerate(((-1, lows), (1, highs))):
            if accelerated is None:
                cell_extrema = np.full(count, -np.inf)
                np.maximum.at(cell_extrema, cross_cells, sign * xy[:, axis])
            else:
                cell_extrema = accelerated[side]
            extreme = _supported_extreme(
                cell_extrema,
                band_mm,
                min_cells,
                quantile=quantile,
                quantile_window_mm=quantile_window_mm,
                preserve_separated_extents=preserve_separated_extents,
            )
            if extreme is not None:
                bounds[axis] = sign * extreme
        if highs[axis] - lows[axis] < min_side_mm:
            lows[axis], highs[axis] = xy[:, axis].min(), xy[:, axis].max()
    return lows, highs


def _supported_height(
    footprint_mm: NDArray[np.float64],
    heights_mm: NDArray[np.float64],
    *,
    voxel_size_mm: float,
    min_height_mm: float,
    band_mm: float,
    min_support_voxels: int,
    quantile: float | None = None,
    quantile_window_mm: float | None = None,
    preserve_separated_extents: bool = False,
) -> float | None:
    maxima = native.height_maxima(
        footprint_mm, heights_mm, voxel_size_mm, min_height_mm
    )
    if maxima is not None:
        if len(maxima) < min_support_voxels:
            return None
        return _supported_extreme(
            maxima,
            band_mm,
            min_support_voxels,
            quantile=quantile,
            quantile_window_mm=quantile_window_mm,
            preserve_separated_extents=preserve_separated_extents,
        )
    above_plane = heights_mm >= min_height_mm
    grid = np.floor(footprint_mm[above_plane] / voxel_size_mm)
    cells, membership = np.unique(grid, axis=0, return_inverse=True)
    if len(cells) < min_support_voxels:
        return None
    cell_maxima = np.full(len(cells), -np.inf)
    np.maximum.at(cell_maxima, membership, heights_mm[above_plane])
    return _supported_extreme(
        cell_maxima,
        band_mm,
        min_support_voxels,
        quantile=quantile,
        quantile_window_mm=quantile_window_mm,
        preserve_separated_extents=preserve_separated_extents,
    )


def _rectangle_axes(angle_deg: float) -> NDArray[np.float64]:
    angle = np.deg2rad(angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, sine], [-sine, cosine]])


def _trimmed_rectangle_angle(
    xy: NDArray[np.float64],
    cell_mm: float,
    tail_fraction: float,
    baseline_angle_deg: float,
) -> float:
    _, membership, counts = np.unique(
        np.floor(xy / cell_mm),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    low_rank = int(len(counts) * tail_fraction)
    high_rank = len(counts) - low_rank - 1
    if low_rank == 0 or high_rank - low_rank < 2:
        return baseline_angle_deg
    centroids = np.column_stack(
        [np.bincount(membership, weights=coordinate) / counts for coordinate in xy.T]
    )

    def trimmed_area(angle_deg: float) -> float:
        projected = centroids @ _rectangle_axes(angle_deg).T
        projected.partition((low_rank, high_rank), axis=0)
        length, width = projected[high_rank] - projected[low_rank]
        return float(length * width)

    coarse_angles = np.r_[baseline_angle_deg, np.arange(0.0, 90.0, 3.0)]
    best = min(coarse_angles, key=trimmed_area)
    fine_angles = np.r_[best, best + np.arange(-2.5, 3.0, 0.5)]
    return float(min(fine_angles, key=trimmed_area))


def measure(
    parcel: PointCloud,
    plane: SupportPlane,
    *,
    outlier_voxel_size_mm: float,
    outlier_radius_mm: float,
    outlier_min_neighbor_voxels: int,
    footprint_edge_band_mm: float,
    min_footprint_edge_cells: int,
    min_footprint_side_mm: float,
    min_height_mm: float,
    height_support_band_mm: float,
    min_height_support_voxels: int,
    footprint_edge_quantile: float | None = None,
    height_cell_quantile: float | None = None,
    orientation_tail_fraction: float = 0.0,
    extent_quantile_window_mm: float | None = None,
    preserve_separated_extents: bool = False,
) -> Cuboid | None:
    if not 0 <= orientation_tail_fraction < 0.5:
        raise ValueError("Orientation tail fraction must be in [0, 0.5)")
    if extent_quantile_window_mm is not None and not extent_quantile_window_mm > 0:
        raise ValueError("Extent quantile window must be positive")
    xyz = parcel.xyz_mm
    if len(xyz) < 3:
        return None

    centroids, point_to_voxel = _voxel_centroids(xyz, outlier_voxel_size_mm)
    counts = cKDTree(centroids).query_ball_point(
        centroids, outlier_radius_mm, return_length=True
    )
    normal = np.asarray(plane.normal)
    tangent = np.eye(3)[np.argmin(np.abs(normal))]
    tangent -= (tangent @ normal) * normal
    tangent /= np.linalg.norm(tangent)
    axes = np.stack((tangent, np.cross(normal, tangent)))
    keep = counts - 1 >= outlier_min_neighbor_voxels
    prepared = native.supported_footprint(
        xyz, point_to_voxel, keep, axes, normal, plane.offset_mm, 1
    )
    if prepared is None:
        supported = xyz[keep[point_to_voxel]]
        footprint = supported @ axes.T
        heights = supported @ normal + plane.offset_mm
    else:
        footprint, heights, origin = prepared
    if len(footprint) < 3:
        return None
    if prepared is None:
        origin = footprint.mean(axis=0)
    _, _, angle_deg = cv2.minAreaRect(
        native.centered_hull_candidates(footprint, origin)
    )
    if orientation_tail_fraction:
        angle_deg = _trimmed_rectangle_angle(
            footprint - origin,
            outlier_voxel_size_mm,
            orientation_tail_fraction,
            angle_deg,
        )
    rectangle_axes = _rectangle_axes(angle_deg)
    extrema = native.rectangle_extrema(
        footprint, origin, rectangle_axes, outlier_voxel_size_mm
    )
    if extrema is None:
        low, high = _footprint_bounds(
            (footprint - origin) @ rectangle_axes.T,
            outlier_voxel_size_mm,
            footprint_edge_band_mm,
            min_footprint_edge_cells,
            min_footprint_side_mm,
            quantile=footprint_edge_quantile,
            quantile_window_mm=extent_quantile_window_mm,
            preserve_separated_extents=preserve_separated_extents,
        )
    else:
        low, high, strips = extrema
        for axis in range(2):
            for sign, bounds, values in (
                (-1, low, strips[axis][0]),
                (1, high, strips[axis][1]),
            ):
                extreme = _supported_extreme(
                    values,
                    footprint_edge_band_mm,
                    min_footprint_edge_cells,
                    quantile=footprint_edge_quantile,
                    quantile_window_mm=extent_quantile_window_mm,
                    preserve_separated_extents=preserve_separated_extents,
                )
                if extreme is not None:
                    bounds[axis] = sign * extreme
            if high[axis] - low[axis] < min_footprint_side_mm:
                # Preserve the original untrimmed fallback for a narrow supported fit.
                aligned = (footprint - origin) @ rectangle_axes.T
                low[axis], high[axis] = aligned[:, axis].min(), aligned[:, axis].max()
    length, width = high - low
    if min(length, width) < min_footprint_side_mm:
        return None

    axis_xy = rectangle_axes[0]
    if width > length:
        length, width = width, length
        axis_xy = rectangle_axes[1]
    length_axis = axis_xy @ axes
    if length_axis[np.argmax(np.abs(length_axis))] < 0:
        length_axis *= -1
    center_xy = ((low + high) / 2) @ rectangle_axes
    base = (center_xy + origin) @ axes - plane.offset_mm * normal
    height = _supported_height(
        footprint,
        heights,
        voxel_size_mm=outlier_voxel_size_mm,
        min_height_mm=min_height_mm,
        band_mm=height_support_band_mm,
        min_support_voxels=min_height_support_voxels,
        quantile=height_cell_quantile,
        quantile_window_mm=extent_quantile_window_mm,
        preserve_separated_extents=preserve_separated_extents,
    )
    return Cuboid(
        base_center_mm=(float(base[0]), float(base[1]), float(base[2])),
        length_axis=(
            float(length_axis[0]),
            float(length_axis[1]),
            float(length_axis[2]),
        ),
        length_mm=float(length),
        width_mm=float(width),
        height_mm=height,
    )
