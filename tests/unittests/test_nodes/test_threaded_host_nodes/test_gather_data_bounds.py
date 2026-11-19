"""Regression coverage for ordered and bounded gathering without a device."""

import unittest
from datetime import timedelta
from unittest.mock import Mock, patch

import depthai as dai

from depthai_nodes.node import GatherData
from depthai_nodes.node import gather_data as gather_module
from tests.utils import OutputMock, PipelineMock


class GatherDataBoundsTests(unittest.TestCase):
    def gather(self, references, data, ticks=None, wait_count_fn=None, **options):
        settings = dict(
            timeout=timedelta(seconds=2), maxPendingFrames=64, preserveOrder=True
        )
        settings.update(options)
        node = (
            PipelineMock()
            .create(GatherData)
            .build(
                cameraFps=25,
                inputData=OutputMock(),
                inputReference=OutputMock(),
                waitCountFn=wait_count_fn,
                **settings,
            )
        )
        node.sent = []
        node.out.send = node.sent.append
        reference_stream, data_stream = iter(references), iter(data)
        node._reference_input.tryGet = lambda: next(reference_stream, None)
        node._data_input.tryGet = lambda: next(data_stream, None)
        iterations = ticks or max(len(references), len(data)) + 1
        node.isRunning = Mock(side_effect=[True] * iterations + [False])
        now = 0

        def advance(_):
            nonlocal now
            now += 0.01

        with patch.object(gather_module, "monotonic", lambda: now), patch.object(
            gather_module, "sleep", advance
        ):
            node.run()
        return node

    def test_default_mode_keeps_ready_first_behavior(self):
        node = self.gather(
            [self.reference(1, 1), self.reference(2, 0)],
            [],
            timeout=None,
            maxPendingFrames=None,
            preserveOrder=False,
        )
        self.assertEqual([group.getSequenceNum() for group in node.sent], [2])

    def test_ordering_can_wait_without_a_timeout(self):
        node = self.gather(
            [self.reference(1, 1), self.reference(2, 0)],
            [],
            ticks=220,
            timeout=None,
            maxPendingFrames=None,
        )
        self.assertFalse(node.sent)
        self.assertEqual(len(node._reference_data), 2)

    def test_expiry_does_not_require_ordered_output(self):
        node = self.gather([self.reference(1, 1)], [], ticks=220, preserveOrder=False)
        self.assertFalse(node._reference_data)
        self.assertFalse(node._arrived_at)

    def test_invalid_limits_are_rejected(self):
        for options in (
            dict(timeout=timedelta(0)),
            dict(timeout=timedelta(seconds=-1)),
            dict(maxPendingFrames=0),
            dict(maxPendingFrames=-1),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.gather([], [], **options)

    def reference(self, sequence, count):
        message = dai.ImgDetections()
        message.detections = [dai.ImgDetection() for _ in range(count)]
        message.setTimestamp(timedelta(seconds=sequence))
        message.setTimestampDevice(timedelta(seconds=sequence))
        message.setSequenceNum(sequence)
        return message

    def result(self, sequence, index=0):
        message = dai.Buffer()
        message.setTimestamp(timedelta(seconds=sequence))
        message.setTimestampDevice(timedelta(seconds=sequence))
        message.setSequenceNum(sequence)
        message.setData([index])
        return message

    def test_zero_one_many_preserve_indices_and_metadata(self):
        references = [self.reference(1, 0), self.reference(2, 1), self.reference(3, 12)]
        data = [None, self.result(2)] + [self.result(3, i) for i in range(12)]
        node = self.gather(references, data)
        self.assertEqual([group.getSequenceNum() for group in node.sent], [1, 2, 3])
        for group, reference in zip(node.sent, references):
            self.assertIs(group.reference_data, reference)
            self.assertEqual(group.getTimestampDevice(), reference.getTimestampDevice())
            self.assertEqual(len(group.items), len(reference.detections))
        self.assertEqual(
            [bytes(item.getData())[0] for item in node.sent[2].items], list(range(12))
        )

    def test_empty_frame_waits_for_delayed_crop(self):
        node = self.gather(
            [self.reference(1, 1), self.reference(2, 0)], [None, None, self.result(1)]
        )
        self.assertEqual([group.getSequenceNum() for group in node.sent], [1, 2])
        self.assertFalse(node._unmatched_data)

    def test_results_can_arrive_before_reference(self):
        node = self.gather(
            [None, None, self.reference(1, 2)], [self.result(1, 0), self.result(1, 1)]
        )
        self.assertEqual(len(node.sent), 1)
        self.assertEqual(
            [bytes(item.getData())[0] for item in node.sent[0].items], [0, 1]
        )

    def test_missing_crop_expires_and_late_result_is_ignored(self):
        data = [self.result(1), self.result(2)] + [None] * 210 + [self.result(1, 1)]
        node = self.gather([self.reference(1, 2), self.reference(2, 1)], data)
        self.assertEqual([group.getSequenceNum() for group in node.sent], [2])
        self.assertFalse(node._reference_data)
        self.assertFalse(node._data_by_reference_ts)
        self.assertFalse(node._unmatched_data)

    def test_final_incomplete_frame_expires_with_idle_inputs(self):
        node = self.gather([self.reference(1, 2)], [self.result(1)], ticks=220)
        self.assertFalse(node.sent)
        self.assertFalse(node._reference_data)
        self.assertFalse(node._arrived_at)

    def test_pending_frame_limit_recovers_before_timeout(self):
        references = [self.reference(1, 1)] + [
            self.reference(i, 0) for i in range(2, 5)
        ]
        node = self.gather(references, [], maxPendingFrames=3)
        self.assertEqual([group.getSequenceNum() for group in node.sent], [2, 3, 4])
        self.assertFalse(node._reference_data)

    def test_results_without_references_are_bounded(self):
        node = self.gather([], [self.result(i) for i in range(100)])
        self.assertEqual(len(node._unmatched_data), node._max_pending_frames)

    def test_empty_decoder_result_still_counts_as_one_crop(self):
        result = dai.MessageGroup()
        result.setTimestamp(timedelta(seconds=1))
        result.setSequenceNum(1)
        node = self.gather([self.reference(1, 1)], [result])
        self.assertEqual(len(node.sent), 1)
        self.assertEqual(node.sent[0].items, [result])

    def test_multi_head_results_keep_face_order(self):
        results = []
        for index in range(2):
            result = dai.MessageGroup()
            result["age"] = self.result(1, index)
            result["gender"] = self.result(1, index + 10)
            result.setTimestamp(timedelta(seconds=1))
            result.setSequenceNum(1)
            results.append(result)
        node = self.gather(
            [self.reference(1, 2), self.reference(2, 0)], [None, None] + results
        )
        self.assertEqual([group.getSequenceNum() for group in node.sent], [1, 2])
        self.assertEqual(node.sent[0].items, results)
        self.assertEqual(
            [bytes(item["gender"].getData())[0] for item in node.sent[0].items],
            [10, 11],
        )

    def test_missing_tile_recovers_with_frame_references(self):
        frames = []
        for sequence in (1, 2):
            frame = dai.ImgFrame()
            frame.setTimestamp(timedelta(seconds=sequence))
            frame.setSequenceNum(sequence)
            frames.append(frame)
        node = self.gather(
            frames,
            [self.result(1)] + [self.result(2, index) for index in range(3)],
            ticks=220,
            wait_count_fn=lambda _: 3,
        )
        self.assertEqual([group.getSequenceNum() for group in node.sent], [2])
        self.assertIs(node.sent[0].reference_data, frames[1])
        self.assertEqual(
            [bytes(item.getData())[0] for item in node.sent[0].items], [0, 1, 2]
        )
