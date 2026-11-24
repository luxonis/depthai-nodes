"""Convert ImageAlign source-camera axial millimeters to target-camera axial depth.

ImageAlign places source Z on target pixels. For an undistorted target ray ``r``
and ``T_source_target = [R | t]``, the required target Z is
``(Z_source - t[2]) / (R[2] dot r)``. Rectified source depth must carry its actual
virtual-camera pose. Native source-camera calibration is not a substitute.
"""

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
from typing import Mapping

import depthai as dai
import numpy as np

from depthai_nodes.message.axial_depth import AxialDepthFrameResult


class AxialDepthPolicy(str, Enum):
    """Explicit declaration of the upstream SDK's axial depth convention."""

    SOURCE_AXIAL_REQUIRES_CORRECTION = "source_axial_requires_correction"
    TARGET_AXIAL_PASSTHROUGH = "target_axial_passthrough"


def _matrix(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {shape} matrix")
    return array


def _rigid(value):
    pose = _matrix(value, (4, 4), "Camera pose")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-7, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Camera pose must be a proper rigid transformation")
    return pose


def _intrinsics(value):
    matrix = _matrix(value, (3, 3), "Intrinsics")
    if np.linalg.det(matrix) == 0 or not np.allclose(
        matrix[2], (0, 0, 1), atol=1e-9, rtol=0
    ):
        raise ValueError("Intrinsics must describe finite pinhole pixel rays")
    return matrix


def _tuples(matrix):
    return tuple(tuple(float(value) for value in row) for row in matrix)


@dataclass(frozen=True)
class AxialDepthGeometry:
    """Immutable geometry for aligned target pixels and source optical Z.

    Attributes:
        target_size: Actual aligned target width and height.
        target_intrinsics: Intrinsics of the undistorted target pixel grid.
        source_from_target_mm: Rigid transform from target optical coordinates
            into the actual source depth optical frame, including rectification.
        identity: Additional immutable geometry identity. ``geometry_from_frames``
            includes both complete image transformations and calibration references.
    """

    target_size: tuple[int, int]
    target_intrinsics: tuple[tuple[float, ...], ...]
    source_from_target_mm: tuple[tuple[float, ...], ...]
    identity: tuple = ()

    def __post_init__(self):
        size = tuple(self.target_size)
        if len(size) != 2 or any(
            not isinstance(value, (int, np.integer)) or value <= 0 for value in size
        ):
            raise ValueError(
                "Target size must contain positive integer width and height"
            )
        object.__setattr__(self, "target_size", tuple(int(value) for value in size))
        object.__setattr__(
            self, "target_intrinsics", _tuples(_intrinsics(self.target_intrinsics))
        )
        object.__setattr__(
            self, "source_from_target_mm", _tuples(_rigid(self.source_from_target_mm))
        )
        identity = tuple(self.identity)
        hash(identity)
        json.dumps(identity, allow_nan=False)
        object.__setattr__(self, "identity", identity)

    @property
    def digest(self) -> str:
        """SHA-256 of all computational geometry and supplied calibration identity."""
        return sha256(
            json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class AxialDepthProvenance:
    """JSON-reportable correction policy, geometry, validity, and association."""

    policy: str
    correction_enabled: bool
    correction_method: str
    input_axial_convention: str
    output_axial_convention: str
    geometry_digest: str
    rejected_valid_source_pixels: int
    sdk_version: str = dai.__version__
    units: str = "mm"
    quantization: str = "Nearest integer millimeter, ties to even"
    invalid_policy: str = "Preserve 0/65535; reject invalid conversions as zero"
    source_association: tuple[tuple[str, int], ...] = ()
    aligned_association: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict:
        """Return JSON-serializable provenance for a benchmark or production report."""
        return asdict(self)


@dataclass(frozen=True)
class AxialDepthResult:
    """Independent target-grid uint16 millimeters and their axial provenance."""

    depth: np.ndarray
    provenance: AxialDepthProvenance


def _association(frame):
    def nanoseconds(value):
        return (
            (value.days * 86400 + value.seconds) * 1000000 + value.microseconds
        ) * 1000

    return (
        ("sequence_num", frame.getSequenceNum()),
        ("timestamp_ns", nanoseconds(frame.getTimestamp())),
        ("timestamp_device_ns", nanoseconds(frame.getTimestampDevice())),
        ("instance_num", frame.getInstanceNum()),
    )


def _raw16_view(frame):
    if (
        not isinstance(frame, dai.ImgFrame)
        or frame.getType() != dai.ImgFrame.Type.RAW16
    ):
        raise TypeError("Expected a RAW16 millimeter ImgFrame")
    width, height = frame.getWidth(), frame.getHeight()
    stride, offset = frame.getStride(), frame.fb.p1Offset
    data = np.asarray(frame.getData(), dtype=np.uint8)
    if (
        width <= 0
        or height <= 0
        or stride < width * 2
        or stride % 2
        or offset < 0
        or offset % 2
        or offset + (height - 1) * stride + width * 2 > data.size
    ):
        raise ValueError("Unsupported or truncated RAW16 row layout")
    return np.ndarray(
        (height, width), dtype="<u2", buffer=data, offset=offset, strides=(stride, 2)
    )


def _describe_geometry(frame):
    transform = frame.getTransformation()
    if transform is None or not transform.isValid():
        raise ValueError("Missing image geometry")
    if tuple(transform.getSize()) != (frame.getWidth(), frame.getHeight()):
        raise ValueError("Image geometry does not match the actual pixel grid")
    source_size = tuple(transform.getSourceSize())
    if len(source_size) != 2 or any(value <= 0 for value in source_size):
        raise ValueError("Missing source image dimensions")
    intrinsic = _intrinsics(transform.getIntrinsicMatrix())
    source_intrinsic = _intrinsics(transform.getSourceIntrinsicMatrix())
    image_matrix = _matrix(transform.getMatrix(), (3, 3), "Image transformation")
    if np.linalg.det(image_matrix) == 0:
        raise ValueError("Image transformation must be invertible")
    coefficients = np.asarray(transform.getDistortionCoefficients(), dtype=np.float64)
    if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
        raise ValueError("Invalid distortion coefficients")
    extrinsics = transform.getExtrinsics()
    if not extrinsics.hasValidRotationMatrix():
        raise ValueError("Missing camera pose")
    pose = _rigid(extrinsics.getTransformationMatrix(unit=dai.LengthUnit.MILLIMETER))
    description = (
        (frame.getWidth(), frame.getHeight()),
        source_size,
        _tuples(intrinsic),
        _tuples(source_intrinsic),
        _tuples(image_matrix),
        transform.getDistortionModel().name,
        tuple(coefficients),
        _tuples(pose),
        extrinsics.toDeviceId,
        int(extrinsics.toCameraSocket),
        frame.getInstanceNum(),
    )
    return transform, description


def geometry_from_frames(
    aligned_frame: dai.ImgFrame, source_frame: dai.ImgFrame
) -> AxialDepthGeometry:
    """Validate exact source association and recover actual optical geometry.

    Args:
        aligned_frame: ImageAlign output on an undistorted perspective target grid.
        source_frame: Corresponding ImageAlign ``passthroughInput``, including the
            source depth's virtual rectified pose.

    Returns:
        Immutable geometry keyed by both full transformations.

    Raises:
        TypeError: If either frame is not RAW16.
        ValueError: If layout, association, calibration, or target projection is invalid.
    """
    _raw16_view(aligned_frame)
    _raw16_view(source_frame)
    if _association(aligned_frame)[:3] != _association(source_frame)[:3]:
        raise ValueError("Aligned and source depth frame association differs")
    target, target_identity = _describe_geometry(aligned_frame)
    source, source_identity = _describe_geometry(source_frame)
    if target.getDistortionModel() != dai.CameraModel.Perspective or np.any(
        np.asarray(target.getDistortionCoefficients()) != 0
    ):
        raise ValueError("Target depth grid must be an undistorted perspective image")
    target_extrinsics, source_extrinsics = (
        target.getExtrinsics(),
        source.getExtrinsics(),
    )
    if (
        target_extrinsics.toDeviceId != source_extrinsics.toDeviceId
        or target_extrinsics.toCameraSocket != source_extrinsics.toCameraSocket
        or target_extrinsics.toCameraSocket == dai.CameraBoardSocket.AUTO
    ):
        raise ValueError(
            "Source and target require the same explicit calibration reference"
        )
    target_pose = np.asarray(
        target_extrinsics.getTransformationMatrix(unit=dai.LengthUnit.MILLIMETER),
        dtype=np.float64,
    )
    source_pose = np.asarray(
        source_extrinsics.getTransformationMatrix(unit=dai.LengthUnit.MILLIMETER),
        dtype=np.float64,
    )
    target_from_source = np.linalg.inv(target_pose) @ source_pose
    pose = np.linalg.inv(target_from_source)
    return AxialDepthGeometry(
        (aligned_frame.getWidth(), aligned_frame.getHeight()),
        target.getIntrinsicMatrix(),
        pose,
        (source_identity, target_identity),
    )


class AxialDepthCorrector:
    """Correct aligned axial Z with a bounded per-instance geometry cache.

    Args:
        policy: Explicit source correction or already-target-axial passthrough.
        cache_entries: Maximum cached denominator grids. Zero disables caching.
        cache_bytes: Maximum bytes retained by cached denominator grids.

    Native frames have no durable arbitrary provenance fields. Retain a returned
    ``AxialDepthFrameResult`` or supply its provenance when crossing this API again.
    """

    def __init__(
        self,
        *,
        policy: AxialDepthPolicy | str,
        cache_entries: int = 4,
        cache_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.policy = AxialDepthPolicy(policy)
        if (
            not isinstance(cache_entries, int)
            or not isinstance(cache_bytes, int)
            or min(cache_entries, cache_bytes) < 0
        ):
            raise ValueError("Cache limits must be nonnegative integers")
        self.cache_entries = cache_entries
        self.cache_bytes = cache_bytes
        self._cache: OrderedDict[AxialDepthGeometry, np.ndarray] = OrderedDict()
        self._cached_bytes = 0

    def _denominator(self, geometry):
        cached = self._cache.get(geometry)
        if cached is not None:
            self._cache.move_to_end(geometry)
            return cached
        width, height = geometry.target_size
        row = np.asarray(geometry.source_from_target_mm)[2]
        coefficients = np.linalg.inv(np.asarray(geometry.target_intrinsics)).T @ row[:3]
        denominator = (
            coefficients[0] * np.arange(width)[None, :]
            + coefficients[1] * np.arange(height)[:, None]
            + coefficients[2]
        )
        denominator.flags.writeable = False
        if self.cache_entries and denominator.nbytes <= self.cache_bytes:
            while self._cache and (
                len(self._cache) >= self.cache_entries
                or self._cached_bytes + denominator.nbytes > self.cache_bytes
            ):
                _, removed = self._cache.popitem(last=False)
                self._cached_bytes -= removed.nbytes
            self._cache[geometry] = denominator
            self._cached_bytes += denominator.nbytes
        return denominator

    def correct(
        self,
        depth: np.ndarray | AxialDepthResult,
        geometry: AxialDepthGeometry,
        *,
        previous_provenance: AxialDepthProvenance | Mapping | None = None,
    ) -> AxialDepthResult:
        """Convert a uint16 target-grid array, or explicitly pass target Z through.

        Args:
            depth: Aligned source axial millimeters, or a previous associated result.
            geometry: Actual source optical pose and undistorted target pixel geometry.
            previous_provenance: Retained provenance when passing a bare array again.

        Returns:
            Independent uint16 array and reportable provenance.

        Raises:
            ValueError: If geometry, shape, or an explicit prior target-Z mark conflicts.
            TypeError: If depth is not a two-dimensional uint16 array.
        """
        if isinstance(depth, AxialDepthResult):
            if previous_provenance is not None:
                raise ValueError(
                    "Supply either an associated result or previous provenance"
                )
            previous_provenance = depth.provenance
            depth = depth.depth
        enabled = self.policy == AxialDepthPolicy.SOURCE_AXIAL_REQUIRES_CORRECTION
        if previous_provenance is not None:
            previous = (
                previous_provenance.to_dict()
                if isinstance(previous_provenance, AxialDepthProvenance)
                else previous_provenance
            )
            if enabled and (
                previous.get("correction_enabled")
                or previous.get("output_axial_convention") == "target_camera_axial_z"
            ):
                raise ValueError(
                    "Cannot correct explicitly marked target-camera axial depth twice"
                )
        if (
            not isinstance(depth, np.ndarray)
            or depth.dtype.kind != "u"
            or depth.dtype.itemsize != 2
            or depth.ndim != 2
        ):
            raise TypeError("Depth must be a two-dimensional uint16 array")
        if not isinstance(geometry, AxialDepthGeometry):
            raise TypeError("Expected AxialDepthGeometry")
        width, height = geometry.target_size
        if depth.shape != (height, width):
            raise ValueError("Depth shape differs from target geometry")
        rejected = 0
        if enabled:
            denominator = self._denominator(geometry)
            source_valid = (depth > 0) & (depth < 65535)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                target_z = np.rint(
                    (depth.astype(np.float64) - geometry.source_from_target_mm[2][3])
                    / denominator
                )
            valid = (
                source_valid
                & np.isfinite(target_z)
                & np.isfinite(denominator)
                & (denominator > 0)
                & (target_z > 0)
                & (target_z < 65535)
            )
            output = np.zeros(depth.shape, dtype="<u2")
            output[depth == 65535] = 65535
            output[valid] = target_z[valid].astype("<u2")
            rejected = int(np.count_nonzero(source_valid & ~valid))
        else:
            output = np.array(depth, dtype="<u2", copy=True)
        provenance = AxialDepthProvenance(
            policy=self.policy.value,
            correction_enabled=enabled,
            correction_method="source_axial_to_target_axial/1"
            if enabled
            else "target_axial_passthrough/1",
            input_axial_convention="source_camera_axial_z"
            if enabled
            else "target_camera_axial_z",
            output_axial_convention="target_camera_axial_z",
            geometry_digest=geometry.digest,
            rejected_valid_source_pixels=rejected,
        )
        return AxialDepthResult(output, provenance)

    def correct_frame(
        self,
        aligned_frame: dai.ImgFrame | AxialDepthFrameResult,
        source_frame: dai.ImgFrame,
        *,
        previous_provenance: AxialDepthProvenance | Mapping | None = None,
    ) -> AxialDepthFrameResult:
        """Return independent RAW16 bytes while preserving all aligned native metadata.

        Args:
            aligned_frame: Native aligned depth or a prior associated result.
            source_frame: Exact ImageAlign source passthrough, with actual depth geometry.
            previous_provenance: Retained mark when passing a native frame again.

        Returns:
            Native frame and provenance in one same-runtime message.
        """
        if isinstance(aligned_frame, AxialDepthFrameResult):
            if previous_provenance is not None:
                raise ValueError(
                    "Supply either an associated result or previous provenance"
                )
            previous_provenance = aligned_frame.provenance
            aligned_frame = aligned_frame.frame
        geometry = geometry_from_frames(aligned_frame, source_frame)
        result = self.correct(
            _raw16_view(aligned_frame),
            geometry,
            previous_provenance=previous_provenance,
        )
        output = aligned_frame.clone()
        data = np.asarray(output.getData(), dtype=np.uint8).copy()
        view = np.ndarray(
            result.depth.shape,
            dtype="<u2",
            buffer=data,
            offset=output.fb.p1Offset,
            strides=(output.getStride(), 2),
        )
        view[:] = result.depth
        output.setData(data)
        provenance = replace(
            result.provenance,
            source_association=_association(source_frame),
            aligned_association=_association(aligned_frame),
        )
        return AxialDepthFrameResult(output, provenance)
