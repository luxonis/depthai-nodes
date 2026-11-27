"""Calibrated image and depth evidence for measurement eligibility."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    distance_transform_edt,
    minimum_filter,
    minimum_filter1d,
    rank_filter,
)

from . import numpy_ops as ops
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
class _OtherDepths:
    nearest: NDArray[np.float32]
    second: NDArray[np.float32]
    owner: NDArray[np.int32]

    @classmethod
    def from_layers(cls, layers):
        nearest = np.full(layers.shape[1:], np.inf, np.float32)
        second = np.full_like(nearest, np.inf)
        owner = np.zeros(nearest.shape, np.int32)
        for label, layer in enumerate(layers):
            closer = layer < nearest
            second = np.minimum(second, np.where(closer, nearest, layer))
            owner = np.where(closer, label, owner)
            nearest = np.minimum(nearest, layer)
        return cls(nearest, second, owner)

    def excluding(self, label):
        return np.where(self.owner == label, self.second, self.nearest)


@dataclass(frozen=True)
class _ViewGrids:
    depth: NDArray[np.float32]
    outside: _OtherDepths
    foreground: _OtherDepths


@dataclass(frozen=True)
class Visibility:
    """Classify proposal visibility across calibrated images with a per-scene cache.

    One clear retained view makes a candidate eligible. Candidates blocked in
    every view are partially visible. Insufficient or conflicting evidence is
    uncertain. Pixel thresholds apply to the delivered image grids, and depth
    thresholds use millimeters. The factory supplies these frozen settings.

    The cache belongs to one component instance. Scenes must not be processed
    concurrently on that instance. Input arrays remain unchanged until all
    projection and validation work for a scene has completed.

    Attributes:
        frame_margin_px: Required image margin around a projected silhouette.
        frame_continuation_px: Maximum edge-to-frame distance checked for continuation.
        boundary_band_px: Maximum pixel offset for probes around a silhouette edge.
        boundary_contrast_window_px: Image window size for local contrast evidence.
        boundary_exterior_px: Exterior image-window width for boundary evidence.
        boundary_contrast_intensity: Minimum image-intensity contrast at an edge.
        boundary_depth_gap_mm: Minimum depth difference supporting a boundary.
        min_edge_support_fraction: Required supported fraction of each silhouette edge.
        depth_cell_px: Pixel side length of a nearest-depth grid cell.
        silhouette_close_radius_cells: Morphological closing radius in grid cells.
        silhouette_simplify_px: Pixel tolerance for polygon approximation.
        min_depth_coverage_fraction: Required observed coverage of the silhouette.
        occlusion_depth_mm: Nearer-foreground gap used to identify occlusion.
        occlusion_min_height_mm: Foreground height for occlusion evidence.
        min_occlusion_cells: Minimum connected nearer-foreground area in grid cells.
        max_hull_gap_fraction: Maximum fraction of convex-hull area missing from the
            observed silhouette. Zero disables recovery.
        projection_stride: Row stride for shared scene projection.
        boundary_probe_stride: Stride factor reducing the number of edge samples.
    """

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
    projection_stride: int = 1
    boundary_probe_stride: int = 1

    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if (
            type(self.boundary_probe_stride) is not int  # noqa: E721 - frozen exact-type check
            or self.boundary_probe_stride < 1
        ):
            raise ValueError("boundary_probe_stride must be a positive integer")

    def begin(
        self, scene, foreground_min_height_mm, pool: ThreadPoolExecutor, *, heights=None
    ):
        """Reset scene state and schedule shared projections on the caller's pool.

        Args:
            scene: Scene whose arrays remain stable until scheduled work completes.
            foreground_min_height_mm: Detection's lower foreground height threshold.
            pool: Executor retained by the caller until all submitted work finishes.
            heights: Optional signed heights aligned with the scene cloud rows.
        """
        self._cache.clear()
        minimum = min(foreground_min_height_mm, self.occlusion_min_height_mm)
        active = (
            ops.height_mask(scene, minimum)
            if heights is None
            else np.asarray(heights >= minimum, dtype=np.uint8)
        )
        included = None
        rows = np.arange(
            0, len(scene.cloud.xyz_mm), self.projection_stride, dtype=np.int32
        )
        points = np.asarray(scene.cloud.xyz_mm[rows], dtype=np.float64)
        foreground = (
            active
            if minimum == self.occlusion_min_height_mm
            else (
                ops.height_mask(scene, self.occlusion_min_height_mm)
                if heights is None
                else np.asarray(heights >= self.occlusion_min_height_mm, dtype=np.uint8)
            )
        )
        futures = {
            id(view): pool.submit(
                ops.project_cells,
                scene,
                view,
                self.depth_cell_px,
                active,
                self.projection_stride,
                (rows, points),
            )
            for view in scene.views
        }
        self._cache.update(
            scene=scene, projections=futures, included=included, foreground=foreground
        )

    def prepare(self, scene: Scene, segments: list[Segment], *, pool=None) -> None:
        """Prepare shared depth grids for one scene's nonoverlapping proposals.

        Matching projections from ``begin`` are reused, and per-view grid work
        completes before cached validation starts. Overlapping segments disable
        the shared cache, so validation uses independent projections instead.
        The caller remains responsible for completing work scheduled by ``begin``.

        Args:
            scene: Scene associated with the supplied segments.
            segments: Proposed source-cloud row sets in candidate order.
            pool: Optional caller-owned executor. An omitted pool creates an executor
                scoped to this preparation call.
        """
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
            foreground = ops.height_mask(scene, self.occlusion_min_height_mm)
        self._cache["foreground"] = foreground

        def finish(view):
            future = projections.get(id(view))
            projected = (
                future.result()
                if future is not None
                else ops.project_cells(scene, view, self.depth_cell_px)
            )
            cell = self.depth_cell_px
            h, w = view.image.shape[:2]
            shape = (len(segments) + 1, (h + cell - 1) // cell, (w + cell - 1) // cell)
            grids = (
                None
                if projected is None
                else ops.accumulate_grids(
                    projected,
                    labels,
                    len(segments) + 1,
                    foreground,
                    shape[1:],
                    included,
                )
            )
            if grids is None:
                depth = np.full(shape, np.inf, np.float32)
                foreground_depth = np.full_like(depth, np.inf)
                xyz = scene.cloud.xyz_mm
                for start in range(0, len(xyz), _CHUNK_POINTS):
                    batch = slice(start, start + _CHUNK_POINTS)
                    cells, z, inside = _projected_cells(view, xyz[batch], cell)
                    owners = labels[batch][inside]
                    np.minimum.at(depth, (owners, *cells), z)
                    fg = foreground[batch][inside].astype(bool)
                    np.minimum.at(
                        foreground_depth,
                        (owners[fg], cells[0][fg], cells[1][fg]),
                        z[fg],
                    )
                grids = depth, foreground_depth
            depth, foreground_depth = grids
            return _ViewGrids(
                depth,
                _OtherDepths.from_layers(depth),
                _OtherDepths.from_layers(foreground_depth),
            )

        if pool is None:
            with ThreadPoolExecutor(max_workers=len(scene.views) or 1) as owned_pool:
                self._cache["views"] = dict(
                    zip(map(id, scene.views), owned_pool.map(finish, scene.views))
                )
        else:
            self._cache["views"] = dict(
                zip(map(id, scene.views), pool.map(finish, scene.views))
            )
        self._cache["projections"] = {}

    def _grids(self, scene, view, label):
        grids = self._cache["views"][id(view)]
        return (
            grids.depth[label],
            grids.outside.excluding(label),
            grids.foreground.excluding(label),
        )

    def validate_candidate(self, scene: Scene, candidate: Segment) -> CandidateDecision:
        """Return measurement eligibility from all retained calibrated views.

        Prepared grids are reused only for the same scene object and an exactly
        matching segment. Other candidates are projected independently. This call
        checks clipping, observed depth coverage, occlusion and edge support.

        Args:
            scene: Scene supplying the cloud, plane and calibrated view pixels.
            candidate: Source-cloud indices of one proposed object.

        Returns:
            ``eligible`` if any view is clear, ``partially_visible`` if every view
            is blocked, or ``uncertain`` otherwise, including scenes with no views.
        """
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
        x0, y0, w0, h0 = cv2.boundingRect(occupied.astype(np.uint8))
        padding = 2 * radius + 1
        left, top = max(0, x0 - padding), max(0, y0 - padding)
        right, bottom = (
            min(shape[1], x0 + w0 + padding),
            min(shape[0], y0 + h0 + padding),
        )
        closed = cv2.morphologyEx(
            occupied[top:bottom, left:right].astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(left, top)
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
            count = max(2, int(length))
            if self.boundary_probe_stride > 1:
                count = max(
                    2,
                    (count + self.boundary_probe_stride - 1)
                    // self.boundary_probe_stride,
                )
            samples = start + np.linspace(0.2, 0.8, count)[:, None] * delta
            if check_frame_continuation and self._continuous_strip_to_frame(
                view.image, samples, outward
            ):
                return False
            depth_evidence = self._depth_boundary(
                parcel_depth, outside_depth, samples, outward
            )
            if np.mean(depth_evidence) >= self.min_edge_support_fraction:
                continue
            image_evidence = np.zeros(len(samples), dtype=bool)
            unresolved = ~depth_evidence
            image_evidence[unresolved] = self._image_boundary(
                view.image, parcel_depth, samples[unresolved], outward
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
        # The 3x3 filter can produce finite neighbors only within one cell of
        # the region. Crop this exact domain before filtering/labeling.
        if observed_region is None:
            x, y, w, h = cv2.boundingRect(np.asarray(region, dtype=np.uint8))
            if not w or not h:
                return False
            crop = (
                slice(max(0, y - 1), min(region.shape[0], y + h + 1)),
                slice(max(0, x - 1), min(region.shape[1], x + w + 1)),
            )
            parcel_depth, outside_depth, region = (
                parcel_depth[crop],
                outside_depth[crop],
                region[crop],
            )
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
        windows = len(offsets) - inner_width - outer_width + 1
        median = rank_filter(values, inner_width // 2, size=(1, inner_width, 1))
        inner = median[:, inner_width // 2 : inner_width // 2 + windows].astype(
            np.float64
        )
        if inner_width % 2 == 0:
            lower = rank_filter(values, inner_width // 2 - 1, size=(1, inner_width, 1))
            inner = (
                inner + lower[:, inner_width // 2 : inner_width // 2 + windows]
            ) / 2
        matches_parcel = (
            np.abs(inner - reference[:, None]).max(axis=2)
            < self.boundary_contrast_intensity
        )
        exterior = np.abs(values.astype(np.float64) - reference[:, None]).max(axis=2)
        minimum = minimum_filter1d(exterior, outer_width, axis=1)
        contrast = (
            minimum[
                :,
                inner_width + outer_width // 2 : inner_width
                + outer_width // 2
                + windows,
            ]
            >= self.boundary_contrast_intensity
        )
        fits = minimum_filter1d(inside, inner_width + outer_width, axis=1)
        fits_image = fits[
            :,
            (inner_width + outer_width) // 2 : (inner_width + outer_width) // 2
            + windows,
        ]
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
        # Keep coordinate operations in their original order, but use contiguous
        # coordinate planes instead of allocating/transposing an N x K x 2 tensor.
        inner = self._probe_band(parcel_depth, samples, outward, -offsets)
        outer = self._probe_band(outside_depth, samples, outward, offsets)
        return (
            np.isfinite(inner)
            & np.isfinite(outer)
            & (outer >= inner + self.boundary_depth_gap_mm)
        )

    def _probe_band(self, grid, samples, outward, offsets):
        x = np.floor(
            (samples[:, 0, None] + offsets[None, :] * outward[0]) / self.depth_cell_px
        ).astype(np.int32)
        y = np.floor(
            (samples[:, 1, None] + offsets[None, :] * outward[1]) / self.depth_cell_px
        ).astype(np.int32)
        height, width = grid.shape
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        values = grid[y.clip(0, height - 1), x.clip(0, width - 1)]
        return np.where(inside, values, np.inf).min(axis=1)

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
