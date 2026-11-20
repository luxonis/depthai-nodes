"""Run native DepthAI parcel checks without repository node monkeypatches."""

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
    ParcelCameraCalibration,
    ParcelCuboid,
    ParcelDetections,
    ParcelMeasurement,
    ParcelMeasurements,
    ParcelPlane,
    ParcelRigCalibration,
    SupportPlane,
)
from depthai_nodes.node import ParcelDetector, ParcelDimensioner
from depthai_nodes.node.parcel_detector.scene import prepare_scene, upward_plane

IDENTITY = tuple(map(tuple, np.eye(4)))
K = ((100, 0, 15.5), (0, 100, 11.5), (0, 0, 1))
PLANE = SupportPlane(normal=(0, 0, -2), offset_mm=2000)


def calibration(size=(32, 24)):
    return ParcelRigCalibration(
        frame="scene",
        down=(0, 0, 1),
        cameras=tuple(
            ParcelCameraCalibration(
                role=role,
                size=size,
                intrinsics=K,
                distortion_model="none",
                distortion_coefficients=(),
                scene_to_camera_mm=IDENTITY,
            )
            for role in ("color", "left", "right")
        ),
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
    supplied = ParcelPlane(frame="scene", plane=plane)
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
        self.detector = self.pipeline.create(ParcelDetector).build(
            *self.source.outputs, calibration()
        )
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
        scene = prepare_scene(inputs[0], inputs[1:4], PLANE, calibration())
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
        inputs[1] = stamp(frame)
        scene = prepare_scene(inputs[0], tuple(inputs[1:4]), PLANE, calibration())
        expected = cv2.cvtColorTwoPlane(
            y, uv.reshape(12, 16, 2), cv2.COLOR_YUV2RGB_NV12
        )
        np.testing.assert_array_equal(scene.views[0].image, expected)

    def test_strided_source_arrays_retain_pixel_order(self):
        inputs = list(messages())
        pixels = np.arange(24 * 64 * 3, dtype=np.uint8).reshape(24, 64, 3)[:, ::2]
        inputs[1].setFrame(pixels)
        scene = prepare_scene(inputs[0], tuple(inputs[1:4]), PLANE, calibration())
        np.testing.assert_array_equal(scene.views[0].image, pixels[:, :, ::-1])

    def test_padded_native_image_rows_preserve_delivered_pixels(self):
        inputs = list(messages())
        for index, channels in ((1, 3), (2, 1), (3, 1)):
            stride = 40 * channels
            pixels = np.full((24, stride), 199, np.uint8)
            pixels[:, : 32 * channels] = np.arange(32 * channels, dtype=np.uint8)
            inputs[index].setStride(stride)
            inputs[index].setData(pixels.ravel())
        scene = prepare_scene(inputs[0], tuple(inputs[1:4]), PLANE, calibration())
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
        scene = prepare_scene(inputs[0], tuple(inputs[1:4]), PLANE, calibration())
        expected = cv2.cvtColorTwoPlane(
            y, uv.reshape(12, 16, 2), cv2.COLOR_YUV2RGB_NV12
        )
        np.testing.assert_array_equal(scene.views[0].image, expected)

    def test_plane_is_snapshotted_normalized_and_flipped_upward(self):
        supplied = ParcelPlane(
            frame="scene", plane=SupportPlane(normal=(0, 0, 3), offset_mm=-3000)
        )
        plane = upward_plane(supplied, calibration())
        self.assertEqual(plane, PLANE)
        self.assertIsNot(plane, supplied.plane)

    def test_distortion_and_transform_reach_the_algorithm_unchanged(self):
        transform = np.eye(4)
        transform[0, 3] = -15
        camera = replace(
            calibration().cameras[0],
            distortion_model="perspective",
            distortion_coefficients=(0.1, 0.01, 0.001, 0.002, 0.003),
            scene_to_camera_mm=transform,
        )
        rig = replace(calibration(), cameras=(camera, *calibration().cameras[1:]))
        inputs = messages()
        scene = prepare_scene(inputs[0], inputs[1:4], PLANE, rig)
        self.assertEqual(
            scene.views[0].distortion_coefficients, camera.distortion_coefficients
        )
        np.testing.assert_array_equal(scene.views[0].scene_to_camera_mm, transform)
        np.testing.assert_array_equal(scene.views[0].intrinsics, camera.intrinsics)

    def test_calibration_snapshots_mutable_inputs(self):
        matrix = np.eye(4)
        intrinsics = np.array(K)
        camera = replace(
            calibration().cameras[0], intrinsics=intrinsics, scene_to_camera_mm=matrix
        )
        matrix[0, 3] = 42
        intrinsics[0, 0] = 42
        self.assertEqual(camera.scene_to_camera_mm[0][3], 0)
        self.assertEqual(camera.intrinsics[0][0], 100)

    def test_bad_calibration_is_rejected(self):
        for field in ({"down": (0, 0, 2)}, {"frame": ""}, {"cameras": ()}):
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                replace(calibration(), **field)
        for field in (
            {"size": (0, 24)},
            {"distortion_model": "unknown"},
            {"distortion_model": "fisheye", "distortion_coefficients": ()},
        ):
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                replace(calibration().cameras[0], **field)


class NodeTests(unittest.TestCase):
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
            inputs[4] = stamp(ParcelPlane(frame="scene", plane=None))
            self.assertEqual(app.detect(inputs).failure_code, "missing_support_plane")

    def test_mismatched_sequence_device_time_and_plane_frame_fail(self):
        with App() as app:
            inputs = list(messages())
            inputs[2].setSequenceNum(9)
            self.assertEqual(app.detect(inputs).failure_code, "unsynchronized_sources")
            inputs = list(messages())
            stamp(inputs[3], device_offset=11)
            self.assertEqual(app.detect(inputs).failure_code, "unsynchronized_sources")
            inputs = list(messages())
            inputs[4] = stamp(ParcelPlane(frame="another", plane=PLANE))
            self.assertEqual(app.detect(inputs).failure_code, "plane_frame_mismatch")

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
                app.detector.build(*app.source.outputs, calibration())
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
            detector = pipeline.create(ParcelDetector).build(
                *source.streams, calibration()
            )
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
                detector = pipeline.create(ParcelDetector).build(
                    *source.streams, calibration()
                )
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
    def test_public_types_and_scoped_statuses(self):
        import depthai_nodes.message as messages
        from depthai_nodes.message import parcel

        public = {
            "Parcel",
            "ParcelCameraCalibration",
            "ParcelCuboid",
            "ParcelDetections",
            "ParcelMeasurement",
            "ParcelMeasurements",
            "ParcelPlane",
            "ParcelRigCalibration",
        }
        self.assertEqual(set(parcel.__all__), public)
        self.assertTrue(public.issubset(messages.__all__))
        for name in (
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
