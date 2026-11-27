"""Vectorized NumPy/OpenCV operations; no custom native library or baseline imports."""
import cv2
import numpy as np

from .scene import project_points


def voxel_centroids(xyz, size, rows=None):
    points = xyz if rows is None else xyz[rows]
    if not len(points):
        return None
    grid = [np.floor(axis / size) for axis in points.T]
    bounds = [(float(axis.min()), float(axis.max())) for axis in grid]
    limit = np.iinfo(np.int64)
    if any(
        not np.isfinite((low, high)).all() or low <= limit.min or high >= limit.max
        for low, high in bounds
    ):
        return None
    span = [int(high) - int(low) + 1 for low, high in bounds]
    if span[0] * span[1] * span[2] > limit.max:
        return None
    grid = [axis.astype(np.int64) - int(low) for axis, (low, high) in zip(grid, bounds)]
    keys = (grid[0] * span[1] + grid[1]) * span[2] + grid[2]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    compressed = len(starts) * 4 < len(points)
    lengths = np.diff(np.r_[starts, len(points)])
    if int(span[0]) * int(span[1]) * int(span[2]) < min(8_000_000, 4 * len(points)):
        dense = (
            np.bincount(keys[starts], weights=lengths).astype(np.int64)
            if compressed
            else np.bincount(keys)
        )
        occupied = dense != 0
        counts = dense[occupied]
        membership = (np.cumsum(occupied) - 1)[keys]
    else:
        _, membership, counts = np.unique(keys, return_inverse=True, return_counts=True)
    if compressed:
        coordinates = np.add.reduceat(
            points.astype(np.float64), starts, axis=0
        ).T.copy()
        members = membership[starts]
    else:
        coordinates = points.T.astype(np.float64, order="C")
        members = membership

    def weighted(axis):
        return np.bincount(members, weights=axis, minlength=len(counts)) / counts

    centers = np.column_stack([weighted(axis) for axis in coordinates])
    return centers, membership


def supported_footprint(xyz, membership, keep, axes, normal, offset):
    supported = xyz[keep[membership]].astype(np.float64)
    if not len(supported):
        return np.empty((0, 2)), np.empty(0), np.zeros(2)
    matrix = np.column_stack((np.vstack((axes, normal)), [0.0, 0.0, offset]))
    projected = cv2.transform(supported.reshape(1, -1, 3), matrix).reshape(-1, 3)
    footprint = projected[:, :2]
    return footprint, projected[:, 2], np.array([axis.mean() for axis in footprint.T])


def cross_extrema(xy, axis, cell):
    keys = np.floor(xy[:, 1 - axis] / cell).astype(np.int64)
    keys -= keys.min()
    if keys.max() > 8_000_000:
        _, keys = np.unique(keys, return_inverse=True)
    count = keys.max() + 1
    lo, hi = np.full(count, -np.inf), np.full(count, -np.inf)
    np.maximum.at(lo, keys, -xy[:, axis])
    np.maximum.at(hi, keys, xy[:, axis])
    used = np.isfinite(hi)
    return lo[used], hi[used]


def height_maxima(xy, heights, cell, min_height):
    above = heights >= min_height
    if not above.any():
        return np.empty(0)
    grid = [np.floor(axis[above] / cell).astype(np.int64) for axis in xy.T]
    grid = [axis - axis.min() for axis in grid]
    span = [int(axis.max()) + 1 for axis in grid]
    if int(span[0]) * int(span[1]) > 8_000_000:
        _, keys = np.unique(np.column_stack(grid), axis=0, return_inverse=True)
    else:
        keys = grid[0] * span[1] + grid[1]
    output = np.full(keys.max() + 1, -np.inf)
    np.maximum.at(output, keys, heights[above])
    return output[np.isfinite(output)]


def threshold_heights(scene, thresholds):
    """Fast float32 classification with conservative float64 boundary refinement.

    Values are used only for comparisons with the supplied thresholds. The bound exceeds
    the accumulated float32 coefficient/product/addition error. Measurement computes its
    own full-precision coordinates independently.
    """
    xyz = scene.cloud.xyz_mm
    if not len(xyz):
        return np.empty(0, dtype=np.float64)
    normal = np.asarray(scene.plane.normal, dtype=np.float64)
    offset = scene.plane.offset_mm
    minimum, maximum, _, _ = cv2.minMaxLoc(xyz)
    bound = 2e-6 * (
        max(abs(minimum), abs(maximum)) * np.abs(normal).sum() + abs(offset)
    )
    coarse = cv2.transform(
        xyz.reshape(1, -1, 3), np.array([[*normal, offset]])
    ).reshape(-1)
    refine = ~np.isfinite(coarse)
    for threshold in thresholds:
        refine |= np.abs(coarse - threshold) <= bound
    result = coarse.astype(np.float64)
    if refine.any():
        points = xyz[refine].astype(np.float64)
        result[refine] = (
            (points[:, 0] * normal[0] + points[:, 1] * normal[1])
            + points[:, 2] * normal[2]
            + offset
        )
    return result


def height_mask(scene, low):
    return np.asarray(
        scene.cloud.xyz_mm @ scene.plane.normal + scene.plane.offset_mm >= low,
        dtype=np.uint8,
    )


def centered_hull_candidates(xy, origin):
    centered = np.float32(xy - origin)
    if len(centered) < 16:
        return centered
    x = np.ascontiguousarray(centered[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(centered[:, 1], dtype=np.float64)
    extrema = centered[
        [
            np.argmax(x),
            np.argmax(y),
            np.argmin(x),
            np.argmin(y),
            np.argmax(x + y),
            np.argmax(x - y),
            np.argmin(x - y),
            np.argmin(x + y),
        ]
    ]
    hull = cv2.convexHull(extrema, clockwise=False)[:, 0].astype(np.float64)
    if len(hull) < 3:
        return centered
    low = np.array([x.min(), y.min()])
    high = np.array([x.max(), y.max()])
    step = (high - low) / 128
    if np.any(step <= 0) or not np.isfinite(step).all():
        return centered
    gx, gy = np.meshgrid(
        low[0] + (np.arange(128) + 0.5) * step[0],
        low[1] + (np.arange(128) + 0.5) * step[1],
    )
    interior = np.ones(gx.shape, dtype=bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        dx, dy = b - a
        # A linear half-plane reaches its minimum at one cell corner.
        # Drop a cell only when all four corners lie strictly inside the hull.
        margin = abs(dx) * (step[1] / 2 + 1e-3) + abs(dy) * (step[0] / 2 + 1e-3)
        interior &= dx * (gy - a[1]) - dy * (gx - a[0]) > margin
    ix = np.clip(((x - low[0]) / step[0]).astype(np.int32), 0, 127)
    iy = np.clip(((y - low[1]) / step[1]).astype(np.int32), 0, 127)
    return centered[~interior[iy, ix]]


def _project(xyz, view):
    """OpenCV perspective model including rational, prism and tilted-sensor terms."""
    model = view.distortion_model.casefold()
    coefficients = view.distortion_coefficients
    if (
        model not in ("none", "perspective")
        or (model == "none" and any(coefficients))
        or len(coefficients) not in (0, 4, 5, 8, 12, 14)
    ):
        return project_points(xyz, view)
    transform = np.asarray(view.scene_to_camera_mm, dtype=np.float64)
    points = np.asarray(xyz, dtype=np.float64)
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    camera = (
        points
        if np.array_equal(transform, np.eye(4))
        else cv2.transform(points.reshape(1, -1, 3), transform[:3]).reshape(-1, 3)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x, y = (camera[:, :2] / camera[:, 2:]).T
        if coefficients and model == "perspective":
            k = np.zeros(14)
            k[: len(coefficients)] = coefficients
            r2 = x * x + y * y
            radial = (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2) / (
                1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2
            )
            xd = (
                x * radial
                + 2 * k[2] * x * y
                + k[3] * (r2 + 2 * x * x)
                + k[8] * r2
                + k[9] * r2 * r2
            )
            yd = (
                y * radial
                + k[2] * (r2 + 2 * y * y)
                + 2 * k[3] * x * y
                + k[10] * r2
                + k[11] * r2 * r2
            )
            if k[12] or k[13]:
                cx, sx = np.cos(k[12]), np.sin(k[12])
                cy, sy = np.cos(k[13]), np.sin(k[13])
                rotation = np.array(
                    [[cy, sy * sx, -sy * cx], [0, cx, sx], [sy, -cy * sx, cy * cx]]
                )
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
                inverse = 1 / (tilt[2, 0] * xd + tilt[2, 1] * yd + tilt[2, 2])
                x = (tilt[0, 0] * xd + tilt[0, 1] * yd + tilt[0, 2]) * inverse
                y = (tilt[1, 0] * xd + tilt[1, 1] * yd + tilt[1, 2]) * inverse
            else:
                x, y = xd, yd
        intr = np.asarray(view.intrinsics, dtype=np.float64)
        z = intr[2, 0] * x + intr[2, 1] * y + intr[2, 2]
        u = (intr[0, 0] * x + intr[0, 1] * y + intr[0, 2]) / z
        v = (intr[1, 0] * x + intr[1, 1] * y + intr[1, 2]) / z
        output = np.column_stack((u, v, camera[:, 2]))
    valid = (
        (camera[:, 2] > 0) & np.isfinite(u) & np.isfinite(v) & np.isfinite(camera[:, 2])
    )
    output[~valid] = np.nan
    return output


def project_cells(scene, view, cell, active=None, stride=1, prepared=None):
    if prepared is None:
        rows = np.arange(0, len(scene.cloud.xyz_mm), stride, dtype=np.int32)
        xyz = scene.cloud.xyz_mm if stride == 1 else scene.cloud.xyz_mm[rows]
    else:
        rows, xyz = prepared
    h, w = view.image.shape[:2]
    shape = ((h + cell - 1) // cell, (w + cell - 1) // cell)
    cells = np.full(len(xyz), -1, np.int32)
    depths = np.empty(len(xyz), np.float32)
    background = np.full(shape, np.inf, np.float32)
    for start in range(0, len(xyz), 131072):
        end = min(start + 131072, len(xyz))
        p = _project(xyz[start:end], view)
        inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        local = np.flatnonzero(inside)
        keys = (p[inside, 1] / cell).astype(np.int32) * shape[1] + (
            p[inside, 0] / cell
        ).astype(np.int32)
        cells[start + local] = keys
        depths[start:end] = p[:, 2]

    return cells, depths, background, rows


def accumulate_grids(projected, labels, label_count, foreground, shape, included=None):
    cells, depths, background, rows = projected
    valid = cells >= 0
    rows = rows[valid]
    cells = cells[valid]
    depths = depths[valid]
    depth = np.full((label_count, *shape), np.inf, np.float32)
    fg = np.full_like(depth, np.inf)
    keys = labels[rows] * int(np.prod(shape)) + cells
    np.minimum.at(depth.ravel(), keys, depths)
    mask = foreground[rows].astype(bool)
    np.minimum.at(fg.ravel(), keys[mask], depths[mask])
    return depth, fg
