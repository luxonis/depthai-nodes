from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, minimum_filter

from . import native
from .scene import (
    CameraView,
    CandidateDecision,
    Scene,
    Segment,
    SupportPlane,
    project_points,
)

ViewDecision = Literal["clear", "blocked", "uncertain"]
_CHUNK_POINTS = 65_536


@dataclass(frozen=True)
class Visibility:
    frame_margin_px: int
    frame_continuation_px: int
    boundary_band_px: int
    boundary_contrast_window_px: int
    boundary_exterior_px: int
    boundary_contrast_intensity: float
    boundary_depth_gap_mm: float
    min_edge_support_fraction: float
    depth_cell_px: int
    silhouette_close_radius_cells: int
    silhouette_simplify_px: float
    min_depth_coverage_fraction: float
    occlusion_depth_mm: float
    occlusion_min_height_mm: float
    min_occlusion_cells: int
    max_hull_gap_fraction: float = 0.0

    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def begin(self, scene, foreground_min_height_mm, pool: ThreadPoolExecutor):
        self._cache.clear()
        minimum = min(foreground_min_height_mm, self.occlusion_min_height_mm)
        active = native.height_mask(scene, minimum)
        included = np.flatnonzero(active).astype(np.int32)
        foreground = (
            active
            if minimum == self.occlusion_min_height_mm
            else native.height_mask(scene, self.occlusion_min_height_mm)
        )
        futures = {
            id(view): pool.submit(
                native.project_cells, scene, view, self.depth_cell_px, active
            )
            for view in scene.views
        }
        self._cache.update(
            scene=scene, projections=futures, included=included, foreground=foreground
        )

    def prepare(self, scene: Scene, segments: list[Segment]) -> None:
        labels = np.zeros(len(scene.cloud.xyz_mm), dtype=np.int32)
        for label, segment in enumerate(segments, 1):
            if np.any(labels[segment.point_indices]):
                self._cache.clear()
                return
            labels[segment.point_indices] = label
        same_scene = self._cache.get("scene") is scene
        projections = self._cache.get("projections", {}) if same_scene else {}
        included = self._cache.get("included") if same_scene else None
        foreground = self._cache.get("foreground") if same_scene else None
        self._cache.clear()
        self._cache.update(
            scene=scene,
            labels=labels,
            segments=segments,
            views={},
            projections=projections,
        )
        if not segments:
            for future in projections.values():
                future.result()
            self._cache["projections"] = {}
            return
        if foreground is None:
            foreground = native.height_mask(scene, self.occlusion_min_height_mm)
        self._cache["foreground"] = foreground

        def finish(view):
            future = projections.get(id(view))
            projected = (
                future.result()
                if future is not None
                else native.project_cells(scene, view, self.depth_cell_px)
            )
            if projected is None:
                return None
            h, w = view.image.shape[:2]
            c = self.depth_cell_px
            return native.accumulate_grids(
                projected,
                labels,
                len(segments) + 1,
                foreground,
                ((h + c - 1) // c, (w + c - 1) // c),
                included,
            )

        with ThreadPoolExecutor(max_workers=len(scene.views) or 1) as pool:
            self._cache["views"] = dict(
                zip(map(id, scene.views), pool.map(finish, scene.views))
            )
        self._cache["projections"] = {}

    def _grids(self, scene, view, label):
        cache = self._cache
        key = id(view)
        if cache["views"].get(key) is None:
            if "foreground" not in cache:
                cache["foreground"] = np.asarray(
                    scene.cloud.xyz_mm @ scene.plane.normal + scene.plane.offset_mm
                    >= self.occlusion_min_height_mm,
                    dtype=np.uint8,
                )
            count = len(cache["segments"]) + 1
            future = cache["projections"].pop(key, None)
            projected = (
                future.result()
                if future is not None
                else native.project_cells(scene, view, self.depth_cell_px)
            )
            cell = self.depth_cell_px
            h, w = view.image.shape[:2]
            grids = (
                None
                if projected is None
                else native.accumulate_grids(
                    projected,
                    cache["labels"],
                    count,
                    cache["foreground"],
                    ((h + cell - 1) // cell, (w + cell - 1) // cell),
                )
            )
            if grids is None:
                cell = self.depth_cell_px
                h, w = view.image.shape[:2]
                shape = (count, (h + cell - 1) // cell, (w + cell - 1) // cell)
                depth, foreground = (
                    np.full(shape, np.inf, np.float32),
                    np.full(shape, np.inf, np.float32),
                )
                xyz = scene.cloud.xyz_mm
                for start in range(0, len(xyz), _CHUNK_POINTS):
                    batch = slice(start, start + _CHUNK_POINTS)
                    cells, z, inside = _projected_cells(view, xyz[batch], cell)
                    owners = cache["labels"][batch][inside]
                    np.minimum.at(depth, (owners, *cells), z)
                    fg = cache["foreground"][batch][inside].astype(bool)
                    np.minimum.at(
                        foreground, (owners[fg], cells[0][fg], cells[1][fg]), z[fg]
                    )
                grids = depth, foreground
            cache["views"][key] = grids
        depth, foreground = cache["views"][key]
        others = np.arange(len(depth)) != label
        return depth[label], depth[others].min(axis=0), foreground[others].min(axis=0)

    def validate_candidate(self, scene: Scene, candidate: Segment) -> CandidateDecision:
        cache = self._cache
        if cache.get("scene") is scene and len(candidate.point_indices):
            label = int(cache["labels"][candidate.point_indices[0]])
            if label and np.array_equal(
                candidate.point_indices, cache["segments"][label - 1].point_indices
            ):
                decisions = []
                for view in scene.views:
                    grids = self._grids(scene, view, label)
                    decision = self._view_decision(
                        view, None, None, None, scene.plane, grids=grids
                    )
                    if decision == "clear":
                        return "eligible"
                    decisions.append(decision)
                return (
                    "partially_visible"
                    if decisions and all(x == "blocked" for x in decisions)
                    else "uncertain"
                )
        return self._validate_uncached(scene, candidate)

    def _validate_uncached(self, scene: Scene, candidate: Segment) -> CandidateDecision:
        xyz = scene.cloud.xyz_mm
        points = xyz[candidate.point_indices]
        member = np.zeros(len(xyz), dtype=bool)
        member[candidate.point_indices] = True
        decisions = []
        for view in scene.views:
            decision = self._view_decision(view, points, xyz, member, scene.plane)
            if decision == "clear":
                return "eligible"
            decisions.append(decision)
        if decisions and all(decision == "blocked" for decision in decisions):
            return "partially_visible"
        return "uncertain"

    def _view_decision(
        self,
        view: CameraView,
        points: NDArray,
        xyz: NDArray,
        member: NDArray,
        plane: SupportPlane,
        *,
        grids=None,
    ) -> ViewDecision:
        height, width = view.image.shape[:2]
        cell = self.depth_cell_px
        shape = ((height + cell - 1) // cell, (width + cell - 1) // cell)
        parcel_depth = (
            _depth_grid(view, points, shape, cell) if grids is None else grids[0]
        )
        occupied = np.isfinite(parcel_depth)
        if not occupied.any():
            return "uncertain"
        radius = self.silhouette_close_radius_cells
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        closed = cv2.morphologyEx(occupied.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        if contour_area == 0:
            return "uncertain"
        x, y, contour_width, contour_height = cv2.boundingRect(contour)
        if (
            x == 0
            or y == 0
            or x + contour_width == shape[1]
            or y + contour_height == shape[0]
        ):
            return "blocked"
        if (
            x * cell < self.frame_margin_px
            or y * cell < self.frame_margin_px
            or (x + contour_width) * cell > width - self.frame_margin_px
            or (y + contour_height) * cell > height - self.frame_margin_px
        ):
            return "uncertain"
        region = cv2.drawContours(
            np.zeros(shape, dtype=np.uint8), [contour], 0, 1, cv2.FILLED
        )
        if np.count_nonzero(
            occupied & region
        ) < self.min_depth_coverage_fraction * np.count_nonzero(region):
            return "uncertain"

        if grids is None:
            outside_depth = np.full(shape, np.inf, dtype=np.float32)
            foreground_depth = np.full_like(outside_depth, np.inf)
            for start in range(0, len(xyz), _CHUNK_POINTS):
                batch = slice(start, start + _CHUNK_POINTS)
                other = xyz[batch][~member[batch]]
                above_plane = (
                    other @ plane.normal + plane.offset_mm
                    >= self.occlusion_min_height_mm
                )
                cells, depth, in_frame = _projected_cells(view, other, cell)
                np.minimum.at(outside_depth, cells, depth)
                foreground = above_plane[in_frame]
                np.minimum.at(
                    foreground_depth,
                    (cells[0][foreground], cells[1][foreground]),
                    depth[foreground],
                )
        else:
            _, outside_depth, foreground_depth = grids
        if self._obstructed(parcel_depth, foreground_depth, region):
            return "blocked"
        if self._edges_supported(view, contour, parcel_depth, outside_depth):
            return "clear"
        if not self.max_hull_gap_fraction:
            return "uncertain"

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        gap_area = hull_area - contour_area
        if not 0 < gap_area <= self.max_hull_gap_fraction * hull_area:
            return "uncertain"
        hull_region = cv2.drawContours(
            np.zeros(shape, dtype=np.uint8), [hull], 0, 1, cv2.FILLED
        )
        if np.count_nonzero(
            occupied & hull_region
        ) < self.min_depth_coverage_fraction * np.count_nonzero(hull_region):
            return "uncertain"
        if self._obstructed(
            parcel_depth, foreground_depth, hull_region, observed_region=region
        ):
            return "uncertain"
        if self._edges_supported(
            view, hull, parcel_depth, outside_depth, check_frame_continuation=True
        ):
            return "clear"
        return "uncertain"

    def _edges_supported(
        self,
        view: CameraView,
        contour: NDArray,
        parcel_depth: NDArray,
        outside_depth: NDArray,
        *,
        check_frame_continuation: bool = False,
    ) -> bool:
        cell = self.depth_cell_px
        polygon = cv2.approxPolyDP(contour, self.silhouette_simplify_px / cell, True)[
            :, 0
        ]
        if len(polygon) < 3:
            return False
        vertices = (polygon + 0.5) * cell
        orientation = np.sign(cv2.contourArea(contour, oriented=True))
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            delta = end - start
            length = np.linalg.norm(delta)
            outward = orientation * np.array([delta[1], -delta[0]]) / length
            # Adjacent edges can cross the same probe near a vertex.
            samples = (
                start + np.linspace(0.2, 0.8, max(2, int(length)))[:, None] * delta
            )
            if check_frame_continuation and self._continuous_strip_to_frame(
                view.image, samples, outward
            ):
                return False
            image_evidence = self._image_boundary(
                view.image, parcel_depth, samples, outward
            )
            depth_evidence = self._depth_boundary(
                parcel_depth, outside_depth, samples, outward
            )
            if (
                np.mean(image_evidence | depth_evidence)
                < self.min_edge_support_fraction
            ):
                return False
        return True

    def _obstructed(
        self,
        parcel_depth: NDArray,
        outside_depth: NDArray,
        region: NDArray,
        *,
        observed_region: NDArray | None = None,
    ) -> bool:
        # A nearer surface at the observed edge can hide the continuation of a split segment.
        neighbors = minimum_filter(
            np.where(region != 0, parcel_depth, np.inf),
            size=3,
            mode="constant",
            cval=np.inf,
        )
        if observed_region is not None:
            gap = (region != 0) & (observed_region == 0)
            unresolved = gap & ~np.isfinite(neighbors) & np.isfinite(outside_depth)
            if unresolved.any():
                observed = (observed_region != 0) & np.isfinite(parcel_depth)
                nearest = distance_transform_edt(
                    ~observed, return_distances=False, return_indices=True
                )
                neighbors[unresolved] = parcel_depth[tuple(nearest[:, unresolved])]
        nearer = np.isfinite(neighbors) & (
            outside_depth + self.occlusion_depth_mm < neighbors
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(nearer.astype(np.uint8))
        return (
            count > 1 and stats[1:, cv2.CC_STAT_AREA].max() >= self.min_occlusion_cells
        )

    def _image_boundary(
        self,
        image: NDArray,
        parcel_depth: NDArray,
        samples: NDArray,
        outward: NDArray,
    ) -> NDArray[np.bool_]:
        accelerated = native.image_boundary(self, image, parcel_depth, samples, outward)
        if accelerated is not None:
            return accelerated
        inner_width = self.boundary_contrast_window_px
        outer_width = self.boundary_exterior_px
        band = self.boundary_band_px
        offsets = np.arange(-band - inner_width, band + outer_width)
        pixels = np.rint(samples[:, None, :] + offsets[None, :, None] * outward).astype(
            np.int32
        )
        x, y = pixels.transpose(2, 0, 1)
        height, width = image.shape[:2]
        margin = self.frame_margin_px
        inside = (
            (x >= margin) & (x < width - margin) & (y >= margin) & (y < height - margin)
        )
        values = image[y.clip(0, height - 1), x.clip(0, width - 1)]
        if image.ndim == 2:
            values = values[:, :, None]
        cell = self.depth_cell_px
        anchor = (offsets <= -cell) & (offsets > -cell - inner_width)
        anchor_pixels = pixels[:, anchor]
        on_parcel = np.isfinite(self._probe_depth(parcel_depth, anchor_pixels))
        reference = np.median(values[:, anchor], axis=1)
        strips = np.lib.stride_tricks.sliding_window_view(
            values, inner_width + outer_width, axis=1
        )
        inner = np.median(strips[..., :inner_width], axis=-1)
        matches_parcel = (
            np.abs(inner - reference[:, None]).max(axis=2)
            < self.boundary_contrast_intensity
        )
        exterior = np.abs(strips[..., inner_width:] - reference[:, None, :, None]).max(
            axis=2
        )
        contrast = exterior.min(axis=-1) >= self.boundary_contrast_intensity
        fits_image = np.lib.stride_tricks.sliding_window_view(
            inside, inner_width + outer_width, axis=1
        ).all(axis=-1)
        separates = np.any(matches_parcel & contrast & fits_image, axis=1)
        continues = self._continues_to_frame(image, samples, outward, reference)
        return on_parcel & separates & ~continues

    def _continues_to_frame(
        self,
        image: NDArray,
        samples: NDArray,
        outward: NDArray,
        parcel_color: NDArray,
    ) -> NDArray[np.bool_]:
        height, width = image.shape[:2]
        exit_px = _frame_exit_distances(image, samples, outward)
        nearby = (exit_px >= self.boundary_contrast_window_px) & (
            exit_px <= self.frame_continuation_px
        )
        # Depth truncation can leave an apparent edge on a print while parcel material reaches the frame.
        offsets = np.floor(exit_px[nearby])[:, None] - np.arange(
            self.boundary_contrast_window_px
        )
        pixels = np.rint(
            samples[nearby, None, :] + offsets[:, :, None] * outward
        ).astype(np.int32)
        x, y = pixels.transpose(2, 0, 1)
        values = image[y.clip(0, height - 1), x.clip(0, width - 1)]
        if image.ndim == 2:
            values = values[:, :, None]
        contrast = np.abs(np.median(values, axis=1) - parcel_color[nearby]).max(axis=1)
        result = np.zeros(len(samples), dtype=bool)
        result[nearby] = contrast < self.boundary_contrast_intensity
        return result

    def _continuous_strip_to_frame(
        self, image: NDArray, samples: NDArray, outward: NDArray
    ) -> bool:
        window = self.boundary_contrast_window_px
        exit_px = _frame_exit_distances(image, samples, outward)
        nearby = (exit_px >= window) & (exit_px <= self.frame_continuation_px)
        if not nearby.any():
            return False

        offsets = np.minimum(
            np.arange(1 - window, self.frame_continuation_px + 2), exit_px[nearby, None]
        )
        pixels = np.rint(
            samples[nearby, None, :] + offsets[:, :, None] * outward
        ).astype(np.int32)
        x, y = pixels.transpose(2, 0, 1)
        height, width = image.shape[:2]
        values = image[y.clip(0, height - 1), x.clip(0, width - 1)]
        if image.ndim == 2:
            values = values[:, :, None]
        continues = np.zeros(len(samples), dtype=bool)
        continues[nearby] = (
            np.ptp(values, axis=1).max(axis=1) < self.boundary_contrast_intensity
        )
        transitions = np.flatnonzero(np.diff(continues, prepend=False, append=False))
        sample_spacing = np.linalg.norm(samples[-1] - samples[0]) / (len(samples) - 1)
        strip_widths = (transitions[1::2] - transitions[::2] - 1) * sample_spacing
        return bool(np.any(strip_widths >= window))

    def _depth_boundary(
        self,
        parcel_depth: NDArray,
        outside_depth: NDArray,
        samples: NDArray,
        outward: NDArray,
    ) -> NDArray[np.bool_]:
        offsets = np.arange((self.depth_cell_px + 1) // 2, self.boundary_band_px + 1)
        if not len(offsets):
            return np.zeros(len(samples), dtype=bool)
        delta = offsets[None, :, None] * outward
        inner = self._probe_depth(parcel_depth, samples[:, None, :] - delta)
        outer = self._probe_depth(outside_depth, samples[:, None, :] + delta)
        return (
            np.isfinite(inner)
            & np.isfinite(outer)
            & (outer >= inner + self.boundary_depth_gap_mm)
        )

    def _probe_depth(self, grid: NDArray, pixels: NDArray) -> NDArray:
        x, y = np.floor(pixels / self.depth_cell_px).astype(np.int32).transpose(2, 0, 1)
        height, width = grid.shape
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        values = grid[y.clip(0, height - 1), x.clip(0, width - 1)]
        return np.where(inside, values, np.inf).min(axis=1)


def _frame_exit_distances(
    image: NDArray, samples: NDArray, outward: NDArray
) -> NDArray:
    height, width = image.shape[:2]
    frame = np.where(outward > 0, (width - 1, height - 1), 0)
    distances = np.full_like(samples, np.inf)
    np.divide(frame - samples, outward, out=distances, where=outward != 0)
    return distances.min(axis=1)


def _depth_grid(
    view: CameraView,
    xyz: NDArray,
    shape: tuple[int, int],
    cell_px: int,
) -> NDArray[np.float32]:
    result = np.full(shape, np.inf, dtype=np.float32)
    for start in range(0, len(xyz), _CHUNK_POINTS):
        cells, depth, _ = _projected_cells(
            view, xyz[start : start + _CHUNK_POINTS], cell_px
        )
        np.minimum.at(result, cells, depth)
    return result


def _projected_cells(
    view: CameraView, xyz: NDArray, cell_px: int
) -> tuple[tuple[NDArray, NDArray], NDArray, NDArray]:
    height, width = view.image.shape[:2]
    projected = project_points(xyz, view)
    x, y, _ = projected.T
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, depth = projected[inside].T
    cells = (
        np.floor(y / cell_px).astype(np.int32),
        np.floor(x / cell_px).astype(np.int32),
    )
    return cells, depth, inside
