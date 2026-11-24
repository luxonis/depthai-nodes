"""Independent projection and message contracts for axial depth correction."""

import json
from dataclasses import replace
from datetime import datetime, timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import AxialDepthFrameResult
from depthai_nodes.utils.axial_depth import (
    AxialDepthCorrector,
    AxialDepthGeometry,
    AxialDepthPolicy,
    geometry_from_frames,
)

CORRECT = AxialDepthPolicy.SOURCE_AXIAL_REQUIRES_CORRECTION
PASS = AxialDepthPolicy.TARGET_AXIAL_PASSTHROUGH


def pose(yaw=0, pitch=0, translation=(0, 0, 0)):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    yaw_rotation = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    pitch_rotation = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    result = np.eye(4)
    result[:3, :3] = yaw_rotation @ pitch_rotation
    result[:3, 3] = translation
    return result


def geometry(size=(9, 7), transform=None, intrinsics=None, identity=()):
    width, height = size
    if intrinsics is None:
        intrinsics = [[20, 0, width / 2], [0, 23, height / 2], [0, 0, 1]]
    return AxialDepthGeometry(
        size, intrinsics, np.eye(4) if transform is None else transform, identity
    )


def make_frame(size=(9, 7), camera_pose=None, *, padded=False, instance=0):
    width, height = size
    stride, offset = (width * 2 + 6, 4) if padded else (width * 2, 0)
    payload = np.full(offset + height * stride + 8, 171, dtype=np.uint8)
    active = np.ndarray(
        (height, width), dtype="<u2", buffer=payload, offset=offset, strides=(stride, 2)
    )
    active[:] = 1000
    frame = dai.ImgFrame()
    frame.setType(dai.ImgFrame.Type.RAW16)
    frame.setWidth(width)
    frame.setHeight(height)
    frame.setStride(stride)
    frame.setData(payload)
    frame.fb.p1Offset = offset
    frame.setTransformation(
        dai.ImgTransformation(
            width,
            height,
            [[20.0, 0, width / 2], [0, 23.0, height / 2], [0, 0, 1]],
            dai.CameraModel.Perspective,
            [0.0] * 5,
            dai.Extrinsics(
                (np.eye(4) if camera_pose is None else camera_pose).tolist(),
                dai.CameraBoardSocket.CAM_A,
                dai.LengthUnit.MILLIMETER,
            ),
        )
    )
    frame.setInstanceNum(instance)
    frame.setSequenceNum(41)
    frame.setTimestamp(timedelta(seconds=2, microseconds=12))
    frame.setTimestampDevice(timedelta(seconds=1, microseconds=19))
    frame.setTimestampSystem(datetime(2026, 11, 24, 12, 0, 0))
    frame.setCategory(8)
    frame.cam.exposureTimeUs = 123
    frame.cam.sensitivityIso = 432
    return frame


def active_pixels(frame):
    return np.ndarray(
        (frame.getHeight(), frame.getWidth()),
        dtype="<u2",
        buffer=frame.getData(),
        offset=frame.fb.p1Offset,
        strides=(frame.getStride(), 2),
    ).copy()


@pytest.mark.parametrize(
    "transform",
    [
        np.eye(4),
        pose(translation=(12, -9, 30)),
        pose(yaw=0.28),
        pose(pitch=-0.22),
        pose(0.31, -0.17, (35, -20, 41)),
    ],
)
def test_recovers_target_z_from_independent_forward_3d_projection(transform):
    geom = geometry(transform=transform)
    width, height = geom.target_size
    k = np.asarray(geom.target_intrinsics)
    target_z = 1200 + np.arange(width * height).reshape(height, width) * 3.25
    pixels = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
    target_xyz = np.empty((height, width, 3))
    target_xyz[..., 0] = (pixels[..., 0] - k[0, 2]) / k[0, 0] * target_z
    target_xyz[..., 1] = (pixels[..., 1] - k[1, 2]) / k[1, 1] * target_z
    target_xyz[..., 2] = target_z
    source_xyz = target_xyz @ transform[:3, :3].T + transform[:3, 3]
    raw = np.rint(source_xyz[..., 2]).astype(np.uint16)
    result = AxialDepthCorrector(policy=CORRECT).correct(raw, geom)
    source_z_per_target_z = (source_xyz[..., 2] - transform[2, 3]) / target_z
    bound = 0.5 + 0.5 / source_z_per_target_z
    assert np.all(np.abs(result.depth.astype(float) - target_z) <= bound + 1e-9)
    if not np.allclose(transform, np.eye(4)):
        assert np.max(np.abs(raw.astype(float) - target_z)) > 5


def test_sentinels_rejections_and_input_ownership():
    raw = np.array([[0, 65535, 1, 50, 2000, 65534]], dtype=np.uint16)
    before = raw.copy()
    result = AxialDepthCorrector(policy=CORRECT).correct(
        raw, geometry((6, 1), pose(translation=(0, 0, 50)))
    )
    np.testing.assert_array_equal(result.depth, [[0, 65535, 0, 0, 1950, 65484]])
    np.testing.assert_array_equal(raw, before)
    assert not np.shares_memory(raw, result.depth)
    assert result.provenance.rejected_valid_source_pixels == 2


@pytest.mark.parametrize("transform", [pose(yaw=np.pi), pose(yaw=np.pi / 2)])
def test_nonpositive_denominators_become_zero(transform):
    geom = geometry((1, 1), transform, np.eye(3))
    result = AxialDepthCorrector(policy=CORRECT).correct(
        np.array([[1000]], np.uint16), geom
    )
    assert result.depth.item() == 0


def test_overflow_and_reserved_saturated_conversion_become_zero():
    geom = geometry((3, 1), pose(translation=(0, 0, -10)))
    result = AxialDepthCorrector(policy=CORRECT).correct(
        np.array([[65525, 65524, 65535]], np.uint16), geom
    )
    np.testing.assert_array_equal(result.depth, [[0, 65534, 65535]])


def test_ties_round_to_even():
    geom = geometry((4, 1), pose(translation=(0, 0, -0.5)))
    result = AxialDepthCorrector(policy=CORRECT).correct(
        np.array([[1, 2, 3, 4]], np.uint16), geom
    )
    np.testing.assert_array_equal(result.depth, [[2, 2, 4, 4]])


def test_explicit_passthrough_and_marked_double_conversion_guard():
    raw = np.array([[0, 1234, 65535]], np.uint16)
    geom = geometry((3, 1), pose(yaw=0.3))
    passthrough = AxialDepthCorrector(policy=PASS).correct(raw, geom)
    np.testing.assert_array_equal(passthrough.depth, raw)
    assert not passthrough.provenance.correction_enabled
    assert passthrough.provenance.input_axial_convention == "target_camera_axial_z"
    json.dumps(passthrough.provenance.to_dict(), allow_nan=False)
    for prior in (passthrough.provenance, passthrough.provenance.to_dict()):
        with pytest.raises(ValueError, match="twice"):
            AxialDepthCorrector(policy=CORRECT).correct(
                raw, geom, previous_provenance=prior
            )
    with pytest.raises(ValueError, match="twice"):
        AxialDepthCorrector(policy=CORRECT).correct(passthrough, geom)
    np.testing.assert_array_equal(
        AxialDepthCorrector(policy=PASS).correct(passthrough, geom).depth, raw
    )


def test_full_geometry_cache_cannot_reuse_same_size_stale_calibration():
    corrector = AxialDepthCorrector(policy=CORRECT, cache_entries=1, cache_bytes=2048)
    raw = np.full((7, 9), 1200, np.uint16)
    first = geometry(transform=pose(0.3))
    second = geometry(transform=pose(0.3), intrinsics=[[7, 0, 1], [0, 8, 2], [0, 0, 1]])
    expected = AxialDepthCorrector(policy=CORRECT, cache_entries=0)
    for geom in (first, second, first, replace(first, identity=("new-calibration",))):
        actual = corrector.correct(raw, geom)
        np.testing.assert_array_equal(actual.depth, expected.correct(raw, geom).depth)
    assert not np.array_equal(
        corrector.correct(raw, first).depth, corrector.correct(raw, second).depth
    )
    assert first.digest != replace(first, identity=("new-calibration",)).digest


@pytest.mark.parametrize("cache_entries,cache_bytes", [(0, 1024), (4, 1)])
def test_cache_limits_do_not_change_results(cache_entries, cache_bytes):
    raw = np.full((7, 9), 1000, np.uint16)
    corrector = AxialDepthCorrector(
        policy=CORRECT, cache_entries=cache_entries, cache_bytes=cache_bytes
    )
    np.testing.assert_array_equal(corrector.correct(raw, geometry()).depth, raw)


def test_geometry_copies_mutable_input_matrices():
    intrinsic, transform = np.eye(3), np.eye(4)
    geom = geometry(intrinsics=intrinsic, transform=transform)
    intrinsic[0, 0] = 100
    transform[2, 3] = 20
    assert geom.target_intrinsics[0][0] == 1
    assert geom.source_from_target_mm[2][3] == 0


@pytest.mark.parametrize(
    "change",
    ["singular_k", "nonfinite_k", "nonrigid_pose", "reflection", "nonfinite_pose"],
)
def test_invalid_geometry_fails_visibly(change):
    intrinsic, transform = np.eye(3), np.eye(4)
    if change == "singular_k":
        intrinsic[0, 0] = 0
    if change == "nonfinite_k":
        intrinsic[0, 0] = np.nan
    if change == "nonrigid_pose":
        transform[0, 0] = 2
    if change == "reflection":
        transform[0, 0] = -1
    if change == "nonfinite_pose":
        transform[0, 3] = np.inf
    with pytest.raises(ValueError):
        geometry(intrinsics=intrinsic, transform=transform)


@pytest.mark.parametrize(
    "raw",
    [np.zeros((7, 9), np.float32), np.zeros((7, 9), np.int16), np.zeros(63, np.uint16)],
)
def test_invalid_depth_array_types_fail(raw):
    with pytest.raises(TypeError):
        AxialDepthCorrector(policy=CORRECT).correct(raw, geometry())


def test_metadata_padded_layout_and_independent_buffers_survive_frame_correction():
    aligned = make_frame(padded=True)
    source = make_frame((12, 8), pose(0.13, -0.07, (20, 5, 30)), instance=1)
    input_bytes = aligned.getData().copy()
    source_bytes = source.getData().copy()
    geom = geometry_from_frames(aligned, source)
    corrected = AxialDepthCorrector(policy=CORRECT).correct_frame(aligned, source)
    expected = AxialDepthCorrector(policy=CORRECT).correct(active_pixels(aligned), geom)
    np.testing.assert_array_equal(active_pixels(corrected.frame), expected.depth)
    np.testing.assert_array_equal(aligned.getData(), input_bytes)
    np.testing.assert_array_equal(source.getData(), source_bytes)
    native = corrected.frame
    assert native.getStride() == aligned.getStride()
    assert native.fb.p1Offset == aligned.fb.p1Offset
    assert native.getCategory() == 8
    assert native.cam.exposureTimeUs == 123
    assert native.cam.sensitivityIso == 432
    for getter in (
        "getSequenceNum",
        "getTimestamp",
        "getTimestampDevice",
        "getTimestampSystem",
        "getInstanceNum",
        "getSourceWidth",
        "getSourceHeight",
    ):
        assert getattr(native, getter)() == getattr(aligned, getter)()
    assert native.getTransformation().isEqualTransformation(aligned.getTransformation())
    mask = np.ones(input_bytes.size, bool)
    for row in range(aligned.getHeight()):
        start = aligned.fb.p1Offset + row * aligned.getStride()
        mask[start : start + aligned.getWidth() * 2] = False
    np.testing.assert_array_equal(native.getData()[mask], input_bytes[mask])
    assert corrected.getSequenceNum() == aligned.getSequenceNum()
    assert dict(corrected.provenance.source_association)["instance_num"] == 1
    assert dict(corrected.provenance.aligned_association)["instance_num"] == 0
    assert isinstance(corrected, AxialDepthFrameResult)
    with pytest.raises(ValueError, match="twice"):
        AxialDepthCorrector(policy=CORRECT).correct_frame(corrected, source)


@pytest.mark.parametrize("field", ["sequence", "timestamp", "device_timestamp"])
def test_wrong_source_association_is_rejected(field):
    aligned, source = make_frame(), make_frame(instance=1)
    if field == "sequence":
        source.setSequenceNum(42)
    if field == "timestamp":
        source.setTimestamp(timedelta(seconds=4))
    if field == "device_timestamp":
        source.setTimestampDevice(timedelta(seconds=4))
    with pytest.raises(ValueError, match="association"):
        geometry_from_frames(aligned, source)


def test_source_virtual_rectification_pose_and_target_grid_are_used():
    source = make_frame((12, 8), pose(0.2, -0.1, (30, 4, -2)), instance=1)
    target = make_frame((9, 7), pose(-0.05, 0.03, (-4, 8, 7)))
    geom = geometry_from_frames(target, source)
    source_pose = np.asarray(
        source.getTransformation()
        .getExtrinsics()
        .getTransformationMatrix(unit=dai.LengthUnit.MILLIMETER)
    )
    target_pose = np.asarray(
        target.getTransformation()
        .getExtrinsics()
        .getTransformationMatrix(unit=dai.LengthUnit.MILLIMETER)
    )
    np.testing.assert_allclose(
        geom.source_from_target_mm, np.linalg.inv(source_pose) @ target_pose, atol=1e-5
    )
    assert geom.target_size == (9, 7)
    assert geom.target_intrinsics == tuple(
        tuple(row) for row in target.getTransformation().getIntrinsicMatrix()
    )


@pytest.mark.parametrize(
    "bad", ["distortion", "fisheye", "grid", "reference", "missing", "layout"]
)
def test_invalid_frame_geometry_is_rejected(bad):
    aligned, source = make_frame(), make_frame(instance=1)
    transform = aligned.getTransformation()
    if bad == "distortion":
        transform.setDistortionCoefficients([0.1, 0, 0, 0, 0])
    if bad == "fisheye":
        transform.setDistortionModel(dai.CameraModel.Fisheye)
    if bad == "grid":
        transform.setSize(2, 2)
    if bad == "reference":
        extrinsic = transform.getExtrinsics()
        extrinsic.toDeviceId = "another-device"
        transform.setExtrinsics(extrinsic)
    if bad == "missing":
        transform = dai.ImgTransformation()
    aligned.setTransformation(transform)
    if bad == "layout":
        aligned.setData(np.zeros(2, np.uint8))
    with pytest.raises(ValueError):
        geometry_from_frames(aligned, source)


def test_policy_is_required_and_unknown_policy_rejected():
    with pytest.raises(TypeError):
        AxialDepthCorrector()
    with pytest.raises(ValueError):
        AxialDepthCorrector(policy="automatic")


def test_float64_pose_composition_preserves_reference_half_mm_quantization():
    source_pose = np.array(
        [
            [
                0.970261568328213,
                -0.01929292519148901,
                0.24128877318025418,
                -43.99676182674705,
            ],
            [0, 0.9968186196402251, 0.07970344746970602, -11.459921247726722],
            [
                -0.24205885446576117,
                -0.0773331919431223,
                0.9671747972308892,
                -33.84199169608161,
            ],
            [0, 0, 0, 1],
        ]
    )
    target_pose = np.array(
        [
            [
                0.9998270511026418,
                -0.002607957440802411,
                0.018413756851395218,
                18.63872791010057,
            ],
            [0, 0.9901187526327467, 0.14023143615100575, -4.982645273402402],
            [
                -0.018597523582330553,
                -0.14020718327874848,
                0.9899475126862253,
                -3.8760997264120345,
            ],
            [0, 0, 0, 1],
        ]
    )
    source = make_frame((80, 60), source_pose, instance=1)
    aligned = make_frame((80, 60), target_pose)
    for frame in (source, aligned):
        transform = dai.ImgTransformation(
            80,
            60,
            [[75.0, 0, 40.0], [0, 76.0, 30.0], [0, 0, 1]],
            dai.CameraModel.Perspective,
            [],
            frame.getTransformation().getExtrinsics(),
        )
        frame.setTransformation(transform)
        frame.setData(np.full((60, 80), 637, np.uint16).view(np.uint8).ravel())
    target = aligned.getTransformation()
    source_transform = source.getTransformation()
    target_from_source = np.linalg.inv(
        np.asarray(
            target.getExtrinsics().getTransformationMatrix(
                unit=dai.LengthUnit.MILLIMETER
            ),
            float,
        )
    ) @ np.asarray(
        source_transform.getExtrinsics().getTransformationMatrix(
            unit=dai.LengthUnit.MILLIMETER
        ),
        float,
    )
    row = np.linalg.inv(target_from_source)[2]
    coefficients = np.linalg.inv(target.getIntrinsicMatrix()).T @ row[:3]
    denominator = (
        coefficients[0] * np.arange(80)[None, :]
        + coefficients[1] * np.arange(60)[:, None]
        + coefficients[2]
    )
    reference = np.rint((637.0 - row[3]) / denominator).astype(np.uint16)
    actual = AxialDepthCorrector(policy=CORRECT).correct_frame(aligned, source).frame
    np.testing.assert_array_equal(active_pixels(actual), reference)
    assert active_pixels(actual)[16, 59] == 569
