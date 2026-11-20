"""Run native DepthAI parcel checks without repository node monkeypatches."""

import ast
import inspect
import unittest
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.algorithms import parcel as algorithm
from depthai_nodes.message import (
    Parcel,
    ParcelCuboid,
    ParcelDetections,
    ParcelMeasurement,
    ParcelMeasurements,
    ParcelPlane,
    SupportPlane,
)
from depthai_nodes.node import ParcelDetector, ParcelDimensioner
from depthai_nodes.node.parcel_detector.scene import (
    check_cloud_configuration,
    prepare_scene,
    upward_plane,
)

IDENTITY = tuple(map(tuple, np.eye(4)))
K = ((100, 0, 15.5), (0, 100, 11.5), (0, 0, 1))
PLANE = SupportPlane(normal=(0, 0, -2), offset_mm=2000)


def metadata(
    size=(32, 24),
    *,
    pose=None,
    unit=dai.LengthUnit.MILLIMETER,
    coefficients=(),
    model=dai.CameraModel.Perspective,
    device="synthetic",
    socket=dai.CameraBoardSocket.CAM_A,
    intrinsics=K,
):
    extrinsics = dai.Extrinsics(
        (np.eye(4) if pose is None else pose).tolist(),
        socket,
        unit,
    )
    extrinsics.toDeviceId = device
    return dai.ImgTransformation(
        *size, intrinsics, model, list(coefficients), extrinsics
    )


def stamp(message, sequence=1, device_offset=0):
    message.setSequenceNum(sequence)
    message.setTimestamp(timedelta(seconds=sequence + 1))
    message.setTimestampDevice(
        timedelta(seconds=sequence + 10, milliseconds=device_offset)
    )
    return message


def messages(sequence=1, points=None, plane=PLANE):
    cloud = dai.PointCloudData()
    if points is None:
        points = np.array([[1, 2, 900], [3, 4, 901], [5, 6, 902]], np.float32)
    cloud.setPoints(np.asarray(points, np.float32))
    color = dai.ImgFrame()
    color.setType(dai.ImgFrame.Type.BGR888i)
    color.setFrame(np.full((24, 32, 3), (3, 7, 19), np.uint8))
    color.setWidth(32)
    color.setHeight(24)
    stereo = []
    for _ in range(2):
        image = dai.ImgFrame()
        image.setType(dai.ImgFrame.Type.GRAY8)
        image.setFrame(np.full((24, 32), 35, np.uint8))
        image.setWidth(32)
        image.setHeight(24)
        stereo.append(image)
    cloud.setTransformation(metadata())
    for image in (color, *stereo):
        image.setTransformation(metadata())
    supplied = ParcelPlane(frame="scene", plane=plane, down=(0, 0, 1))
    return tuple(stamp(m, sequence) for m in (cloud, color, *stereo, supplied))


class Source(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.outputs = [self.createOutput(f"source_{n}") for n in range(5)]

    def run(self):
        pass


class App:
    def __init__(self):
        self.pipeline = dai.Pipeline(False)
        self.pipeline.__enter__()
        self.source = self.pipeline.create(Source)
        self.detector = self.pipeline.create(ParcelDetector).build(*self.source.outputs)
        self.dimensioner = self.pipeline.create(ParcelDimensioner).build(
            self.detector.out
        )
        self.detected = self.detector.out.createOutputQueue()
        self.measured = self.dimensioner.out.createOutputQueue()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.pipeline.__exit__(*args)

    def detect(self, inputs=None):
        self.detector.process(*(inputs or messages()))
        return self.detected.get()

    def measure(self, batch):
        self.dimensioner.process(batch)
        return self.measured.get()


def candidate(indices=(2, 0), decision="eligible"):
    return algorithm.CandidateResult(
        algorithm.Segment(np.array(indices, np.int32)), decision
    )


def cuboid(height=100):
    return algorithm.Cuboid((0, 0, 1000), (1, 0, 0), 100, 50, height)


class ConversionTests(unittest.TestCase):
    def test_bgr_is_rgb_and_images_cloud_are_read_only(self):
        inputs = messages()
        scene = prepare_scene(inputs[0], inputs[1:4], PLANE, "scene")
        np.testing.assert_array_equal(scene.views[0].image[0, 0], [19, 7, 3])
        self.assertEqual([v.role for v in scene.views], ["color", "left", "right"])
        self.assertEqual(scene.cloud.xyz_mm.dtype, np.float32)
        self.assertFalse(scene.cloud.xyz_mm.flags.writeable)
        self.assertTrue(all(not v.image.flags.writeable for v in scene.views))
        np.testing.assert_array_equal(scene.cloud.xyz_mm, inputs[0].getPoints())

    def test_nv12_decodes_like_frozen_two_plane_decoder(self):
        inputs = list(messages())
        frame = dai.ImgFrame()
        frame.setType(dai.ImgFrame.Type.NV12)
        y = np.arange(32 * 24, dtype=np.uint8).reshape(24, 32)
        uv = np.full((12, 32), 140, dtype=np.uint8)
        frame.setWidth(32)
        frame.setHeight(24)
        frame.setData(np.concatenate((y.ravel(), uv.ravel())))
        frame.setTransformation(metadata())
        inputs[1] = stamp(frame)
        scene = prepare_scene(inputs[0], tuple(inputs[1:4]), PLANE, "scene")
        expected = cv2.cvtColorTwoPlane(
            y, uv.reshape(12, 16, 2), cv2.COLOR_YUV2RGB_NV12
        )
        np.testing.assert_array_equal(scene.views[0].image, expected)

    def test_strided_source_arrays_retain_pixel_order(self):
        inputs = list(messages())
        pixels = np.arange(24 * 64 * 3, dtype=np.uint8).reshape(24, 64, 3)[:, ::2]
        inputs[1].setFrame(pixels)
        scene = prepare_scene(inputs[0], tuple(inputs[1:4]), PLANE, "scene")
        np.testing.assert_array_equal(scene.views[0].image, pixels[:, :, ::-1])

    def test_padded_native_image_rows_preserve_delivered_pixels(self):
        inputs = list(messages())
        for index, channels in ((1, 3), (2, 1), (3, 1)):
            stride = 40 * channels
            pixels = np.full((24, stride), 199, np.uint8)
            pixels[:, : 32 * channels] = np.arange(32 * channels, dtype=np.uint8)
            inputs[index].setStride(stride)
            inputs[index].setData(pixels.ravel())
        scene = prepare_scene(inputs[0], tuple(inputs[1:4]), PLANE, "scene")
        expected_color = np.tile(np.arange(96, dtype=np.uint8), (24, 1)).reshape(
            24, 32, 3
        )
        np.testing.assert_array_equal(scene.views[0].image, expected_color[:, :, ::-1])
        expected_mono = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
        for view in scene.views[1:]:
            np.testing.assert_array_equal(view.image, expected_mono)
        y = np.arange(32 * 24, dtype=np.uint8).reshape(24, 32)
        uv = np.full((12, 32), 140, np.uint8)
        padded_y = np.pad(y, ((0, 0), (0, 8)), constant_values=199)
        padded_uv = np.pad(uv, ((0, 0), (0, 8)), constant_values=199)
        inputs[1].setType(dai.ImgFrame.Type.NV12)
        inputs[1].setStride(40)
        inputs[1].setData(np.concatenate((padded_y.ravel(), padded_uv.ravel())))
        scene = prepare_scene(inputs[0], tuple(inputs[1:4]), PLANE, "scene")
        expected = cv2.cvtColorTwoPlane(
            y, uv.reshape(12, 16, 2), cv2.COLOR_YUV2RGB_NV12
        )
        np.testing.assert_array_equal(scene.views[0].image, expected)

    def test_plane_is_snapshotted_normalized_and_flipped_upward(self):
        supplied = ParcelPlane(
            frame="scene",
            plane=SupportPlane(normal=(0, 0, 3), offset_mm=-3000),
            down=(0, 0, 1),
        )
        plane = upward_plane(supplied)
        self.assertEqual(plane, PLANE)
        self.assertIsNot(plane, supplied.plane)

    def test_delivered_intrinsics_follow_native_crop_and_scale(self):
        inputs = messages()
        transformation = metadata(size=(64, 48))
        transformation.addCrop(8, 6, 32, 24)
        transformation.addScale(0.5, 0.5)
        inputs[1].setFrame(np.zeros((12, 16, 3), np.uint8))
        inputs[1].setWidth(16)
        inputs[1].setHeight(12)
        inputs[1].setTransformation(transformation)
        scene = prepare_scene(inputs[0], inputs[1:4], PLANE, "scene")
        expected = np.asarray(transformation.getIntrinsicMatrix())
        np.testing.assert_array_equal(scene.views[0].intrinsics, expected)
        self.assertFalse(
            np.array_equal(expected, transformation.getSourceIntrinsicMatrix())
        )
        self.assertEqual(scene.views[0].image.shape, (12, 16, 3))

    def test_distortion_and_pose_use_native_metadata_in_millimeters(self):
        inputs = messages()
        angle = np.deg2rad(17)
        pose = np.array(
            [
                [np.cos(angle), 0, np.sin(angle), 1.5],
                [0, 1, 0, -0.25],
                [-np.sin(angle), 0, np.cos(angle), 0.75],
                [0, 0, 0, 1],
            ]
        )
        transformation = metadata(
            pose=pose,
            unit=dai.LengthUnit.CENTIMETER,
            coefficients=(0.1, 0.01, 0.001, 0.002, 0.003),
        )
        inputs[1].setTransformation(transformation)
        scene = prepare_scene(inputs[0], inputs[1:4], PLANE, "scene")
        expected = np.linalg.inv(
            np.asarray(
                transformation.getExtrinsics().getTransformationMatrix(
                    False, dai.LengthUnit.MILLIMETER
                ),
                dtype=np.float64,
            )
        )
        np.testing.assert_array_equal(scene.views[0].scene_to_camera_mm, expected)
        self.assertEqual(
            scene.views[0].distortion_coefficients,
            tuple(transformation.getDistortionCoefficients()),
        )
        np.testing.assert_array_equal(
            scene.views[0].intrinsics, transformation.getIntrinsicMatrix()
        )

    def test_cloud_source_pose_is_not_applied_twice(self):
        inputs = messages()
        pose = np.eye(4)
        pose[0, 3] = 125
        inputs[0].setTransformation(metadata(pose=pose))
        scene = prepare_scene(inputs[0], inputs[1:4], PLANE, "reference")
        np.testing.assert_array_equal(scene.cloud.xyz_mm, inputs[0].getPoints())
        np.testing.assert_array_equal(scene.views[0].scene_to_camera_mm, np.eye(4))
        self.assertEqual(scene.cloud.frame, "reference")
        self.assertEqual(scene.plane.frame, "reference")

    def test_native_projection_matches_frozen_projector_with_distortion(self):
        for model, coefficients in (
            (
                dai.CameraModel.Perspective,
                (0.01, -0.005, 0.001, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, -0.02),
            ),
            (dai.CameraModel.Fisheye, (0.01, -0.005, 0.001, 0.002)),
        ):
            with self.subTest(model=model):
                inputs = messages()
                transformation = metadata(coefficients=coefficients, model=model)
                inputs[1].setTransformation(transformation)
                scene = prepare_scene(inputs[0], inputs[1:4], PLANE, "scene")
                actual = algorithm.project_points(scene.cloud.xyz_mm, scene.views[0])
                points = scene.cloud.xyz_mm.astype(np.float64)
                distortion = np.asarray(
                    transformation.getDistortionCoefficients(), np.float64
                )
                if model == dai.CameraModel.Perspective:
                    expected, _ = cv2.projectPoints(
                        points,
                        np.zeros(3),
                        np.zeros(3),
                        np.asarray(transformation.getIntrinsicMatrix()),
                        distortion,
                    )
                else:
                    normalized = points[:, :2] / points[:, 2:]
                    expected = cv2.fisheye.distortPoints(
                        normalized.reshape(-1, 1, 2),
                        np.asarray(transformation.getIntrinsicMatrix()),
                        distortion,
                    )
                np.testing.assert_allclose(
                    actual[:, :2], expected.reshape(-1, 2), atol=1e-9, rtol=0
                )

    def test_unavailable_and_invalid_metadata_fail_frames(self):
        with App() as app:
            for index in range(4):
                with self.subTest(index=index):
                    inputs = messages()
                    inputs[index].setTransformation(dai.ImgTransformation())
                    self.assertEqual(
                        app.detect(inputs).failure_code, "missing_camera_metadata"
                    )
            inputs = messages()
            inputs[1].setTransformation(dai.ImgTransformation(32, 24))
            self.assertEqual(app.detect(inputs).failure_code, "missing_camera_metadata")
            inputs = messages()
            inputs[1].setTransformation(metadata(coefficients=(float("nan"), 0, 0, 0)))
            self.assertEqual(app.detect(inputs).failure_code, "invalid_camera_metadata")
            inputs = messages()
            inputs[1].setTransformation(metadata(model=dai.CameraModel.Equirectangular))
            self.assertEqual(
                app.detect(inputs).failure_code, "unsupported_distortion_model"
            )
            for field in (
                {"device": "another"},
                {"socket": dai.CameraBoardSocket.CAM_B},
            ):
                inputs = messages()
                inputs[2].setTransformation(metadata(**field))
                self.assertEqual(
                    app.detect(inputs).failure_code, "incompatible_camera_frames"
                )

    def test_plane_gravity_is_validated_and_snapshotted(self):
        for gravity in ((0, 0, 2), (0, float("nan"), 1), (0, 1)):
            with self.subTest(gravity=gravity), self.assertRaises(ValueError):
                ParcelPlane(frame="scene", plane=PLANE, down=gravity)
        gravity = np.array([0, 0, 1.0])
        supplied = ParcelPlane(frame="scene", plane=PLANE, down=gravity)
        gravity[:] = 0
        self.assertEqual(supplied.down, (0, 0, 1))

    def test_scene_snapshots_native_metadata_before_later_changes(self):
        inputs = messages()
        scene = prepare_scene(inputs[0], inputs[1:4], PLANE, "scene")
        expected = scene.views[0]
        inputs[1].setTransformation(metadata(pose=np.diag([-1.0, -1.0, 1.0, 1.0])))
        np.testing.assert_array_equal(expected.scene_to_camera_mm, np.eye(4))
        np.testing.assert_array_equal(expected.intrinsics, K)


class NodeTests(unittest.TestCase):
    def test_native_generator_configuration_uses_reference_coordinates(self):
        check_cloud_configuration(dai.PointCloudConfig())
        camera = dai.PointCloudConfig().setTargetCoordinateSystem(
            dai.CameraBoardSocket.CAM_A
        )
        centimeters = dai.PointCloudConfig().setLengthUnit(dai.LengthUnit.CENTIMETER)
        transform = np.eye(4)
        transform[0, 3] = 15
        shifted = dai.PointCloudConfig().setTransformationMatrix(transform.tolist())
        for config in (camera, centimeters, shifted):
            with self.subTest(config=config), self.assertRaises(ValueError):
                check_cloud_configuration(config)

    def test_completed_candidate_call_then_read_only_measurement(self):
        events = []
        with App() as app:

            def find(scene):
                events.append("find")
                self.assertFalse(scene.cloud.xyz_mm.flags.writeable)
                return [
                    candidate(decision=d)
                    for d in (
                        "background",
                        "partially_visible",
                        "uncertain",
                        "eligible",
                        "eligible",
                        "eligible",
                    )
                ]

            outcomes = iter((None, cuboid(None), cuboid()))
            app.detector._components = SimpleNamespace(find_candidates=find)
            detected = app.detect()

            def measure(parcel, plane):
                self.assertEqual(events[0], "find")
                events.append("measure")
                self.assertFalse(parcel.xyz_mm.flags.writeable)
                self.assertIs(parcel.xyz_mm, detected.items[len(events) - 2].xyz_mm)
                np.testing.assert_array_equal(
                    parcel.xyz_mm, messages()[0].getPoints()[[2, 0]]
                )
                self.assertEqual(plane.normal, detected.plane.normal)
                self.assertEqual(plane.offset_mm, detected.plane.offset_mm)
                return next(outcomes)

            app.dimensioner._components = SimpleNamespace(measure=measure)
            measured = app.measure(detected)
            self.assertEqual(events, ["find", "measure", "measure", "measure"])
            self.assertEqual([p.candidate_id for p in detected.items], [3, 4, 5])
            self.assertEqual([p.candidate_id for p in measured.items], [3, 4, 5])
            self.assertEqual(
                [p.status for p in measured.items], list(ParcelMeasurement.Status)
            )
            self.assertEqual(len(detected.decisions), 6)

    def test_source_timestamps_and_sequence_survive_both_nodes(self):
        with App() as app:
            app.detector._components = SimpleNamespace(find_candidates=lambda _: [])
            inputs = messages(17)
            detected = app.detect(inputs)
            measured = app.measure(detected)
            for result in (detected, measured):
                self.assertEqual(result.getSequenceNum(), 17)
                self.assertEqual(result.getTimestamp(), inputs[0].getTimestamp())
                self.assertEqual(
                    result.getTimestampDevice(), inputs[0].getTimestampDevice()
                )
                self.assertEqual(
                    [s.stream for s in result.sources],
                    ["point_cloud", "color", "left", "right", "support_plane"],
                )
                self.assertEqual(result.frame, "scene")
                self.assertEqual(result.algorithm_id, algorithm.ALGORITHM_ID)
            self.assertIs(measured.plane, detected.plane)
            self.assertIs(measured.sources, detected.sources)

    def test_empty_and_rejected_frames_are_successful(self):
        for results in ([], [candidate(decision="uncertain")]):
            with self.subTest(results=results), App() as app:
                app.detector._components = SimpleNamespace(
                    find_candidates=lambda _, results=results: results
                )
                detected = app.detect()
                measured = app.measure(detected)
                for result in (detected, measured):
                    self.assertEqual(result.outcome, ParcelDetections.Outcome.SUCCESS)
                    self.assertEqual(result.items, ())
                    self.assertIsNone(result.failure_code)

    def test_empty_scene_calls_real_algorithm(self):
        with App() as app:
            detected = app.detect(messages(points=np.empty((0, 3), np.float32)))
            self.assertEqual(detected.outcome, ParcelDetections.Outcome.SUCCESS)
            self.assertEqual(app.measure(detected).items, ())

    def test_missing_plane_and_each_missing_image_fail(self):
        with App() as app:
            for index in range(1, 5):
                inputs = list(messages())
                inputs[index] = None
                detected = app.detect(inputs)
                self.assertEqual(detected.outcome, ParcelDetections.Outcome.FAILED)
                self.assertEqual(
                    app.measure(detected).outcome, ParcelDetections.Outcome.FAILED
                )
            inputs = list(messages())
            inputs[4] = stamp(ParcelPlane(frame="scene", plane=None, down=(0, 0, 1)))
            self.assertEqual(app.detect(inputs).failure_code, "missing_support_plane")

    def test_mismatched_sequence_device_time_and_camera_reference_fail(self):
        with App() as app:
            inputs = list(messages())
            inputs[2].setSequenceNum(9)
            self.assertEqual(app.detect(inputs).failure_code, "unsynchronized_sources")
            inputs = list(messages())
            stamp(inputs[3], device_offset=11)
            self.assertEqual(app.detect(inputs).failure_code, "unsynchronized_sources")
            inputs = list(messages())
            inputs[2].setTransformation(metadata(device="another"))
            self.assertEqual(
                app.detect(inputs).failure_code, "incompatible_camera_frames"
            )

    def test_grid_format_and_nonfinite_cloud_fail(self):
        with App() as app:
            inputs = list(messages())
            inputs[1].setFrame(np.zeros((12, 16, 3), np.uint8))
            inputs[1].setWidth(16)
            inputs[1].setHeight(12)
            self.assertEqual(
                app.detect(inputs).failure_code, "calibration_grid_mismatch"
            )
            inputs = list(messages())
            inputs[2].setType(dai.ImgFrame.Type.RAW8)
            self.assertEqual(app.detect(inputs).failure_code, "invalid_stereo_format")
            inputs = messages(points=np.array([[0, np.nan, 900]], np.float32))
            self.assertEqual(app.detect(inputs).failure_code, "invalid_point_cloud")

    def test_bad_candidates_and_indices_fail_even_when_rejected(self):
        invalid = [None, (), [None], [algorithm.CandidateResult(None, "eligible")]]
        for indices in (
            [],
            np.array([], np.int32),
            np.array([0], np.int64),
            np.array([[0]], np.int32),
            np.array([-1], np.int32),
            np.array([3], np.int32),
            np.array([0, 0], np.int32),
            np.array([2, 0, 2], np.int32),
        ):
            invalid.append(
                [algorithm.CandidateResult(algorithm.Segment(indices), "background")]
            )
        invalid.append([candidate(decision="unknown")])
        with App() as app:
            for results in invalid:
                with self.subTest(results=results):
                    app.detector._components = SimpleNamespace(
                        find_candidates=lambda _, results=results: results
                    )
                    self.assertEqual(app.detect().failure_code, "detection_failed")

    def test_callback_exceptions_fail_whole_frame(self):
        with App() as app:

            def fail(*_):
                raise RuntimeError("test callback error")

            app.detector._components = SimpleNamespace(find_candidates=fail)
            self.assertEqual(app.detect().outcome, ParcelDetections.Outcome.FAILED)
            app.detector._components = SimpleNamespace(
                find_candidates=lambda _: [candidate()]
            )
            app.dimensioner._components = SimpleNamespace(measure=fail)
            measured = app.measure(app.detect())
            self.assertEqual(measured.failure_code, "measurement_failed")
            self.assertEqual(measured.items, ())

    def test_invalid_cuboid_geometry_fails(self):
        invalid = [
            replace(cuboid(), length_axis=(2, 0, 0)),
            replace(cuboid(), length_axis=(0, 0, 1)),
            replace(cuboid(), base_center_mm=(0, 0, 1001)),
            replace(cuboid(), width_mm=0),
            replace(cuboid(), height_mm=float("nan")),
            1,
        ]
        with App() as app:
            app.detector._components = SimpleNamespace(
                find_candidates=lambda _: [candidate()]
            )
            batch = app.detect()
            for geometry in invalid:
                with self.subTest(geometry=geometry):
                    app.dimensioner._components = SimpleNamespace(
                        measure=lambda *_, geometry=geometry: geometry
                    )
                    self.assertEqual(
                        app.measure(batch).failure_code, "measurement_failed"
                    )

    def test_algorithm_identity_mismatch_is_failed_not_empty(self):
        with App() as app:
            batch = ParcelDetections(
                frame="scene", plane=PLANE, algorithm_id="algorithm-other"
            )
            result = app.measure(batch)
            self.assertEqual(result.failure_code, "algorithm_identity_mismatch")
            self.assertEqual(result.algorithm_id, "algorithm-other")
            self.assertEqual(result.outcome, ParcelDetections.Outcome.FAILED)

    def test_queued_parcels_and_planes_survive_subsequent_frames(self):
        with App() as app:
            app.detector._components = SimpleNamespace(
                find_candidates=lambda _: [candidate()]
            )
            old = app.detect()
            expected = old.items[0].xyz_mm.copy()
            later_plane = SupportPlane(normal=(0, 0, -1), offset_mm=1200)
            later = app.detect(
                messages(2, points=np.full((3, 3), 800), plane=later_plane)
            )
            self.assertEqual(later.plane.offset_mm, 1200)
            seen = []

            def measure(parcel, plane):
                seen.append((parcel.xyz_mm.copy(), plane.offset_mm))
                return None

            app.dimensioner._components = SimpleNamespace(measure=measure)
            app.measure(old)
            np.testing.assert_array_equal(old.items[0].xyz_mm, expected)
            np.testing.assert_array_equal(seen[0][0], expected)
            self.assertEqual(seen[0][1], 1000)
            self.assertFalse(
                np.shares_memory(old.items[0].xyz_mm, later.items[0].xyz_mm)
            )

    def test_public_parcel_constructor_snapshots_points(self):
        points = np.ones((3, 3), np.float32)
        parcel = Parcel(candidate_id=0, xyz_mm=points)
        points[:] = 0
        np.testing.assert_array_equal(parcel.xyz_mm, np.ones((3, 3)))
        self.assertFalse(parcel.xyz_mm.flags.writeable)

    def test_owned_parcel_points_are_retained_without_a_second_copy(self):
        points = np.ones((3, 3), np.float32)
        parcel = Parcel._from_owned(0, points)
        self.assertIs(parcel.xyz_mm, points)
        self.assertFalse(points.flags.writeable)
        with self.assertRaises(AttributeError):
            parcel.candidate_id = 2
        with self.assertRaises(AttributeError):
            parcel.xyz_mm = np.zeros((3, 3), np.float32)

    def test_cuboid_constructor_snapshots_vectors_with_read_only_properties(self):
        center = np.array([0.0, 0.0, 1000.0])
        axis = [1.0, 0.0, 0.0]
        geometry = ParcelCuboid(
            base_center_mm=center,
            length_axis=axis,
            length_mm=100,
            width_mm=50,
            height_mm=None,
        )
        center[:] = 0
        axis[:] = [0, 0, 1]
        self.assertEqual(geometry.base_center_mm, (0, 0, 1000))
        self.assertEqual(geometry.length_axis, (1, 0, 0))
        with self.assertRaises(AttributeError):
            geometry.length_mm = 200

    def test_message_invariants(self):
        with self.assertRaises(ValueError):
            ParcelDetections(frame="scene", plane=None)
        with self.assertRaises(ValueError):
            ParcelMeasurements(
                frame="scene", plane=PLANE, outcome=ParcelDetections.Outcome.FAILED
            )
        with self.assertRaises(ValueError):
            ParcelMeasurement(
                candidate_id=0,
                status=ParcelMeasurement.Status.NO_FIT,
                cuboid=ParcelCuboid(
                    base_center_mm=(0, 0, 0),
                    length_axis=(1, 0, 0),
                    length_mm=100,
                    width_mm=50,
                    height_mm=10,
                ),
            )
        with self.assertRaises(ValueError):
            ParcelDetections(
                frame="scene",
                plane=PLANE,
                items=(Parcel(candidate_id=0, xyz_mm=np.ones((1, 3))),) * 2,
            )

    def test_rebuild_rejected_and_nodes_share_binding(self):
        with App() as app:
            self.assertEqual(app.detector.algorithm_id, app.dimensioner.algorithm_id)
            with self.assertRaises(RuntimeError):
                app.detector.build(*app.source.outputs)
            with self.assertRaises(RuntimeError):
                app.dimensioner.build(app.detector.out)


class MissingSource(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.streams = tuple(self.createOutput(f"s{i}") for i in range(5))
        self.omit = 3
        self.unavailable_plane = False

    def run(self):
        for sequence in (1, 2):
            for index, (output, message) in enumerate(
                zip(
                    self.streams,
                    messages(sequence, plane=None)
                    if sequence == 1 and self.unavailable_plane
                    else messages(sequence),
                )
            ):
                if sequence == 1 and index == self.omit:
                    continue
                output.send(message)


class StreamingTests(unittest.TestCase):
    def test_explicit_unavailable_plane_is_a_failed_frame_then_recovers(self):
        with dai.Pipeline(False) as pipeline:
            source = pipeline.create(MissingSource)
            source.omit = None
            source.unavailable_plane = True
            detector = pipeline.create(ParcelDetector).build(*source.streams)
            dimensioner = pipeline.create(ParcelDimensioner).build(detector.out)
            results = dimensioner.out.createOutputQueue()
            pipeline.start()
            failed = results.get(timedelta(seconds=5))
            recovered = results.get(timedelta(seconds=5))
            self.assertIsNotNone(failed)
            self.assertIsNotNone(recovered)
            self.assertEqual(failed.getSequenceNum(), 1)
            self.assertEqual(failed.outcome, ParcelDetections.Outcome.FAILED)
            self.assertEqual(failed.failure_code, "missing_support_plane")
            self.assertEqual(recovered.getSequenceNum(), 2)
            self.assertEqual(recovered.outcome, ParcelDetections.Outcome.SUCCESS)
            pipeline.stop()

    def test_native_sync_skips_incomplete_groups_and_recovers(self):
        for omitted in (1, 2, 3, 4):
            with self.subTest(omitted=omitted), dai.Pipeline(False) as pipeline:
                source = pipeline.create(MissingSource)
                source.omit = omitted
                detector = pipeline.create(ParcelDetector).build(*source.streams)
                dimensioner = pipeline.create(ParcelDimensioner).build(detector.out)
                results = dimensioner.out.createOutputQueue()
                pipeline.start()
                result = results.get(timedelta(seconds=5))
                self.assertIsNotNone(result)
                self.assertEqual(result.getSequenceNum(), 2)
                self.assertEqual(result.outcome, ParcelDetections.Outcome.SUCCESS)
                self.assertEqual(result.items, ())
                self.assertIsNone(results.get(timedelta(milliseconds=50)))
                pipeline.stop()


class PublicApiTests(unittest.TestCase):
    def test_parcel_values_do_not_bypass_attribute_assignment(self):
        from depthai_nodes.message import parcel

        source = ast.parse(inspect.getsource(parcel))
        forbidden = [
            node
            for node in ast.walk(source)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "object"
            and node.func.attr == "__setattr__"
        ]
        self.assertEqual(forbidden, [])

    def test_public_types_and_scoped_statuses(self):
        import depthai_nodes.message as messages
        from depthai_nodes.message import parcel

        public = {
            "Parcel",
            "ParcelCuboid",
            "ParcelDetections",
            "ParcelMeasurement",
            "ParcelMeasurements",
            "ParcelPlane",
        }
        self.assertEqual(set(parcel.__all__), public)
        self.assertTrue(public.issubset(messages.__all__))
        for name in (
            "ParcelCameraCalibration",
            "ParcelRigCalibration",
            "ParcelDecision",
            "ParcelSourceTiming",
            "ParcelFrameOutcome",
            "ParcelMeasurementStatus",
        ):
            self.assertFalse(hasattr(messages, name))
            self.assertFalse(hasattr(parcel, name))
        self.assertIs(ParcelDetections.Outcome, ParcelMeasurements.Outcome)
        self.assertEqual(ParcelDetections.Outcome.SUCCESS.value, "success")
        self.assertEqual(
            ParcelMeasurement.Status.HEIGHT_UNKNOWN.value, "height_unknown"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
