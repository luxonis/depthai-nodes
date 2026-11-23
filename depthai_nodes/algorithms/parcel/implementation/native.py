"""Optional native kernels; conservative NumPy/OpenCV fallbacks live in
geometry/visibility."""
import ctypes
import sys
from pathlib import Path

import numpy as np

_path = Path(__file__).with_name(
    "_kernels.dylib" if sys.platform == "darwin" else "_kernels.so"
)
try:
    _lib = ctypes.CDLL(str(_path))
except OSError:
    _lib = None


def _pointer(array):
    return array.ctypes.data_as(ctypes.c_void_p)


if _lib is not None:
    _lib.voxel_centroids.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _lib.voxel_centroids.restype = ctypes.c_int
    _lib.height_maxima.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
    ]
    _lib.height_maxima.restype = ctypes.c_int
    _lib.project_cells.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        *[ctypes.c_void_p] * 4,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        *[ctypes.c_void_p] * 4,
    ]
    _lib.project_cells.restype = None
    _lib.accumulate_grids.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int64,
        *[ctypes.c_void_p] * 4,
    ]
    _lib.accumulate_grids.restype = None


def voxel_centroids(xyz, size, rows=None):
    if (
        _lib is None
        or xyz.dtype != np.float32
        or not xyz.flags.c_contiguous
        or not len(xyz)
    ):
        return None
    n = len(xyz) if rows is None else len(rows)
    centroids = np.empty((n, 3), dtype=np.float64)
    membership = np.empty(n, dtype=np.int64)
    count = _lib.voxel_centroids(
        _pointer(xyz),
        n,
        size,
        _pointer(centroids),
        _pointer(membership),
        None if rows is None else _pointer(rows),
    )
    return (centroids[:count], membership) if count >= 0 else None


def height_maxima(xy, heights, cell, min_height):
    if _lib is None or xy.dtype != np.float64 or heights.dtype != np.float64:
        return None
    xy, heights = np.ascontiguousarray(xy), np.ascontiguousarray(heights)
    output = np.empty(len(heights), dtype=np.float64)
    count = _lib.height_maxima(
        _pointer(xy),
        _pointer(heights),
        len(heights),
        cell,
        min_height,
        _pointer(output),
    )
    return output[:count] if count >= 0 else None


def project_cells(scene, view, cell, active=None):
    xyz = scene.cloud.xyz_mm
    if (
        _lib is None
        or xyz.dtype != np.float32
        or not xyz.flags.c_contiguous
        or cell <= 0
    ):
        return None
    model = view.distortion_model.casefold()
    coefficients = view.distortion_coefficients
    if not (
        (model == "perspective" and len(coefficients) in (0, 4, 5, 8, 12, 14))
        or (model == "none" and not any(coefficients))
    ):
        return None
    k = np.zeros(14)
    if model != "none":
        k[: len(coefficients)] = coefficients
    cx, sx = np.cos(k[12]), np.sin(k[12])
    cy, sy = np.cos(k[13]), np.sin(k[13])
    rotation = np.array([[cy, sy * sx, -sy * cx], [0, cx, sx], [sy, -cy * sx, cy * cx]])
    tilt = (
        np.array(
            [
                [rotation[2, 2], 0, -rotation[0, 2]],
                [0, rotation[2, 2], -rotation[1, 2]],
                [0, 0, 1],
            ]
        )
        @ rotation
    )
    t = np.asarray(view.scene_to_camera_mm, dtype=np.float64)
    intr = np.asarray(view.intrinsics, dtype=np.float64)
    h, w = view.image.shape[:2]
    cells, depths = np.empty(len(xyz), np.int32), np.empty(len(xyz), np.float32)
    background = (
        None
        if active is None
        else np.full(
            ((h + cell - 1) // cell, (w + cell - 1) // cell), np.inf, np.float32
        )
    )
    _lib.project_cells(
        _pointer(xyz),
        len(xyz),
        _pointer(t),
        _pointer(k),
        _pointer(intr),
        _pointer(tilt),
        w,
        h,
        cell,
        _pointer(cells),
        _pointer(depths),
        None if active is None else _pointer(active),
        None if background is None else _pointer(background),
    )
    return (cells, depths) if background is None else (cells, depths, background)


def accumulate_grids(projected, labels, label_count, foreground, shape, included=None):
    cells, depths = projected[:2]
    background = projected[2] if len(projected) == 3 else None
    grids, fg = (
        np.empty((label_count, *shape), np.float32),
        np.empty((label_count, *shape), np.float32),
    )
    _lib.accumulate_grids(
        _pointer(cells),
        _pointer(depths),
        len(cells) if included is None else len(included),
        _pointer(labels),
        label_count,
        _pointer(foreground),
        shape[0] * shape[1],
        _pointer(grids),
        _pointer(fg),
        None if included is None else _pointer(included),
        None if background is None else _pointer(background),
    )
    return grids, fg


if _lib is not None:
    _lib.cross_extrema.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _lib.cross_extrema.restype = ctypes.c_int
    _lib.foreground_rows.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
    ]
    _lib.foreground_rows.restype = ctypes.c_int
    _lib.group_rows.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    _lib.group_rows.restype = None


def cross_extrema(xy, axis, cell):
    if _lib is None or xy.dtype != np.float64:
        return None
    xy = np.ascontiguousarray(xy)
    lo, hi = np.empty(len(xy)), np.empty(len(xy))
    count = _lib.cross_extrema(
        _pointer(xy), len(xy), axis, cell, _pointer(lo), _pointer(hi)
    )
    return (lo[:count], hi[:count]) if count >= 0 else None


def foreground_rows(xyz, plane, low, high):
    if _lib is None or xyz.dtype != np.float32 or not xyz.flags.c_contiguous:
        return None
    normal = np.asarray(plane.normal, dtype=np.float64)
    rows = np.empty(len(xyz), dtype=np.int32)
    count = _lib.foreground_rows(
        _pointer(xyz),
        len(xyz),
        _pointer(normal),
        plane.offset_mm,
        low,
        high,
        _pointer(rows),
    )
    return rows[:count]


def group_rows(rows, labels, accepted, counts):
    if _lib is None:
        return None
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    starts = np.where(
        accepted, np.cumsum(np.where(accepted, counts, 0)) - counts, -1
    ).astype(np.int64)
    output = np.empty(int(counts[accepted].sum()), dtype=np.int32)
    _lib.group_rows(
        _pointer(rows),
        _pointer(labels),
        len(rows),
        _pointer(starts),
        len(starts),
        _pointer(output),
    )
    return output


if _lib is not None:
    _lib.image_boundary.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_void_p,
    ]
    _lib.image_boundary.restype = None


def image_boundary(visibility, image, grid, samples, outward):
    if (
        _lib is None
        or image.dtype != np.uint8
        or grid.dtype != np.float32
        or not grid.flags.c_contiguous
        or visibility.boundary_contrast_window_px > 64
        or visibility.boundary_contrast_window_px < 1
        or visibility.boundary_exterior_px < 1
        or visibility.boundary_band_px < visibility.depth_cell_px
    ):
        return None
    channels = 1 if image.ndim == 2 else image.shape[2]
    if channels not in (1, 3):
        return None
    samples, outward = (
        np.ascontiguousarray(samples, dtype=np.float64),
        np.ascontiguousarray(outward, dtype=np.float64),
    )
    params = np.array(
        [
            visibility.depth_cell_px,
            visibility.frame_margin_px,
            visibility.boundary_contrast_window_px,
            visibility.boundary_exterior_px,
            visibility.boundary_band_px,
            visibility.frame_continuation_px,
        ],
        np.int32,
    )
    result = np.empty(len(samples), dtype=np.uint8)
    _lib.image_boundary(
        _pointer(image),
        *image.shape[:2],
        channels,
        image.strides[0],
        image.strides[1],
        image.strides[2] if image.ndim == 3 else 0,
        _pointer(grid),
        *grid.shape,
        _pointer(samples),
        len(samples),
        _pointer(outward),
        _pointer(params),
        visibility.boundary_contrast_intensity,
        _pointer(result),
    )
    return result.astype(bool)


if _lib is not None:
    _lib.height_mask.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
    ]
    _lib.height_mask.restype = None


def height_mask(scene, low):
    xyz = scene.cloud.xyz_mm
    if _lib is None or xyz.dtype != np.float32 or not xyz.flags.c_contiguous:
        return np.asarray(
            xyz @ scene.plane.normal + scene.plane.offset_mm >= low, dtype=np.uint8
        )
    normal = np.asarray(scene.plane.normal, dtype=np.float64)
    mask = np.empty(len(xyz), dtype=np.uint8)
    _lib.height_mask(
        _pointer(xyz),
        len(xyz),
        _pointer(normal),
        scene.plane.offset_mm,
        low,
        _pointer(mask),
    )
    return mask


if _lib is not None:
    _lib.hull_candidates.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    _lib.hull_candidates.restype = ctypes.c_int


def hull_candidates(xy):
    if _lib is None or xy.dtype != np.float32 or not xy.flags.c_contiguous:
        return xy
    output = np.empty_like(xy)
    count = _lib.hull_candidates(_pointer(xy), len(xy), _pointer(output))
    return output[:count] if count >= 3 else xy


if _lib is not None:
    _lib.supported_footprint.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        *[ctypes.c_void_p] * 4,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _lib.supported_footprint.restype = ctypes.c_int


def supported_footprint(xyz, membership, keep, axes, normal, offset, fma_mode=0):
    if _lib is None or xyz.dtype != np.float32 or not xyz.flags.c_contiguous:
        return None
    keep = np.asarray(keep, dtype=np.uint8)
    membership = np.asarray(membership, dtype=np.int64)
    axes, normal = (
        np.ascontiguousarray(axes, dtype=np.float64),
        np.ascontiguousarray(normal, dtype=np.float64),
    )
    xy = np.empty((len(xyz), 2))
    heights = np.empty(len(xyz))
    origin = np.empty(2)
    count = _lib.supported_footprint(
        _pointer(xyz),
        len(xyz),
        _pointer(membership),
        _pointer(keep),
        _pointer(axes),
        _pointer(normal),
        offset,
        _pointer(xy),
        _pointer(heights),
        _pointer(origin),
        fma_mode,
    )
    return xy[:count], heights[:count], origin


if _lib is not None:
    _lib.centered_hull_candidates.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    _lib.centered_hull_candidates.restype = ctypes.c_int
    _lib.rectangle_extrema.argtypes = [
        *[ctypes.c_void_p] * 3,
        ctypes.c_int64,
        ctypes.c_double,
        *[ctypes.c_void_p] * 6,
    ]
    _lib.rectangle_extrema.restype = ctypes.c_int


def centered_hull_candidates(xy, origin):
    if _lib is None or xy.dtype != np.float64 or not xy.flags.c_contiguous:
        return hull_candidates(np.float32(xy - origin))
    output = np.empty(xy.shape, dtype=np.float32)
    count = _lib.centered_hull_candidates(
        _pointer(xy), _pointer(origin), len(xy), _pointer(output)
    )
    return output[:count]


def rectangle_extrema(xy, origin, axes, cell):
    if _lib is None or xy.dtype != np.float64 or not xy.flags.c_contiguous:
        return None
    outputs = [np.empty(len(xy)) for _ in range(4)]
    bounds = np.empty(4)
    counts = np.empty(2, np.int32)
    status = _lib.rectangle_extrema(
        _pointer(xy),
        _pointer(origin),
        _pointer(axes),
        len(xy),
        cell,
        *map(_pointer, outputs),
        _pointer(bounds),
        _pointer(counts),
    )
    if status < 0:
        return None
    return (
        bounds[:2],
        bounds[2:],
        [
            (outputs[2 * i][: counts[i]], outputs[2 * i + 1][: counts[i]])
            for i in range(2)
        ],
    )


if _lib is not None:
    _lib.detect_voxels.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_float,
        *[ctypes.c_void_p] * 4,
    ]
    _lib.detect_voxels.restype = ctypes.c_int


def detect_voxels(scene, low, high, size):
    xyz = scene.cloud.xyz_mm
    if _lib is None or xyz.dtype != np.float32 or not xyz.flags.c_contiguous:
        return None
    normal = np.asarray(scene.plane.normal, dtype=np.float64)
    centroids = np.empty((len(xyz), 3))
    membership = np.empty(len(xyz), np.int64)
    rows = np.empty(len(xyz), np.int32)
    count = np.empty(1, np.int64)
    voxels = _lib.detect_voxels(
        _pointer(xyz),
        len(xyz),
        _pointer(normal),
        scene.plane.offset_mm,
        low,
        high,
        size,
        _pointer(centroids),
        _pointer(membership),
        _pointer(rows),
        _pointer(count),
    )
    return (
        (rows[: count[0]], centroids[:voxels], membership[: count[0]])
        if voxels >= 0
        else None
    )
