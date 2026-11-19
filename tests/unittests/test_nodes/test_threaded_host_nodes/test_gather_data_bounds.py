from datetime import timedelta
from unittest.mock import Mock, patch

import depthai as dai
import pytest

from depthai_nodes.node import GatherData
from depthai_nodes.node import gather_data as gather_module
from tests.utils import OutputMock, PipelineMock


def message(sequence, count=None):
    item = dai.Buffer() if count is None else dai.ImgDetections()
    if count is not None:
        item.detections = [dai.ImgDetection() for _ in range(count)]
    item.setTimestamp(timedelta(seconds=sequence))
    item.setTimestampDevice(timedelta(seconds=sequence, microseconds=7))
    item.setSequenceNum(sequence)
    return item


def gather(references=(), data=(), ticks=600, fps=25, **options):
    node = (
        PipelineMock()
        .create(GatherData)
        .build(
            cameraFps=fps,
            inputData=OutputMock(),
            inputReference=OutputMock(),
            **options,
        )
    )
    references, data = iter(references), iter(data)
    node._reference_input.tryGet = lambda: next(references, None)
    node._data_input.tryGet = lambda: next(data, None)
    node.isRunning = Mock(side_effect=[True] * ticks + [False])
    sent = []
    node.out.send = lambda group: sent.append((clock, group))
    clock = 0

    def advance(delay):
        nonlocal clock
        clock += delay

    with patch.object(gather_module.time, "monotonic", lambda: clock), patch.object(
        gather_module.time, "sleep", advance
    ):
        node.run()
    return node, sent


@pytest.mark.parametrize("preserve_order, expected", [(False, [2]), (True, [1, 2])])
def test_ordering_is_optional(preserve_order, expected):
    _, sent = gather(
        [message(1, 1), message(2, 0)],
        [None, None, message(1)],
        preserveOrder=preserve_order,
    )
    assert [group.getSequenceNum() for _, group in sent] == expected


@pytest.mark.parametrize("fps", [13, 15, 25])
def test_timeout_releases_later_frames_and_ignores_late_data(fps):
    timeout = timedelta(seconds=5 / fps)
    node, sent = gather(
        [message(1, 2), message(2, 1)],
        [message(1), message(2)] + [None] * 550 + [message(1)],
        fps=fps,
        timeout=timeout,
        preserveOrder=True,
    )
    assert [group.getSequenceNum() for _, group in sent] == [2]
    assert timeout.total_seconds() <= sent[0][0] <= 5 / fps + 2 / fps / 100
    assert not node._reference_data
    assert not node._data_by_reference_ts
    assert not node._unmatched_data


@pytest.mark.parametrize("preserve_order", [False, True])
def test_final_incomplete_group_expires_with_idle_inputs(preserve_order):
    node, sent = gather(
        [message(1, 2)],
        [message(1)],
        timeout=timedelta(seconds=0.1),
        preserveOrder=preserve_order,
    )
    assert not sent
    assert not node._reference_data
    assert not node._data_by_reference_ts
    assert not node._arrived_at


def test_ordering_without_limits_waits_for_missing_results():
    node, sent = gather([message(1, 1), message(2, 0)], preserveOrder=True)
    assert not sent
    assert len(node._reference_data) == 2


def test_pending_limit_recovers_without_timeout():
    node, sent = gather(
        [message(1, 1), message(2, 0), message(3, 0)],
        maxPendingFrames=2,
        preserveOrder=True,
    )
    assert [group.getSequenceNum() for _, group in sent] == [2, 3]
    assert not node._reference_data


def test_unmatched_messages_are_bounded():
    node, _ = gather(data=[message(i) for i in range(100)], maxPendingFrames=3)
    assert len(node._unmatched_data) == 3
    assert [item.getSequenceNum() for item in node._unmatched_data] == [97, 98, 99]


def test_zero_one_many_groups_preserve_items_and_metadata():
    references = [message(1, 0), message(2, 1), message(3, 12)]
    data = [message(2)] + [message(3) for _ in range(12)]
    _, sent = gather(references, data, preserveOrder=True)
    assert [group.getSequenceNum() for _, group in sent] == [1, 2, 3]
    assert [len(group.items) for _, group in sent] == [0, 1, 12]
    for (_, group), reference in zip(sent, references):
        assert group.reference_data is reference
        assert group.getTimestamp() == reference.getTimestamp()
        assert group.getTimestampDevice() == reference.getTimestampDevice()
    assert sent[-1][1].items == data[1:]


def test_data_before_reference_and_custom_count():
    reference = message(1)
    data = [message(1), message(1)]
    _, sent = gather(
        [None, None, reference], data, waitCountFn=lambda _: 2, preserveOrder=True
    )
    assert len(sent) == 1
    assert sent[0][1].items == data


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": timedelta(0)},
        {"timeout": timedelta(seconds=-1)},
        {"maxPendingFrames": 0},
        {"maxPendingFrames": -1},
    ],
)
def test_nonpositive_limits_are_rejected(options):
    with pytest.raises(ValueError):
        gather(**options)
