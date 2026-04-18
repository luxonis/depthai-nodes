from datetime import timedelta

import depthai as dai

from depthai_nodes import Collection
from depthai_nodes.node import MessageCollector
from tests.utils import PipelineMock


def create_buffer(sequence_num: int, timestamp_s: float, timestamp_device_s: float):
    msg = dai.Buffer()
    msg.setSequenceNum(sequence_num)
    msg.setTimestamp(timedelta(seconds=timestamp_s))
    msg.setTimestampDevice(timedelta(seconds=timestamp_device_s))
    return msg


def test_run_uses_last_collected_message_metadata():
    collector = PipelineMock().create(MessageCollector)
    collector.setCameraFps(30)

    first_msg = create_buffer(10, 1.000, 11.000)
    second_msg = create_buffer(11, 1.010, 11.010)
    next_group_msg = create_buffer(12, 1.100, 11.100)

    def check(result, *_):
        assert isinstance(result, Collection)
        assert result.items == [first_msg, second_msg]
        assert result.getSequenceNum() == second_msg.getSequenceNum()
        assert result.getTimestamp() == second_msg.getTimestamp()
        assert result.getTimestampDevice() == second_msg.getTimestampDevice()
        collector.setIsRunning(False)

    collector.out.createOutputQueue(checking_function=check)
    collector._data_input.send(first_msg)
    collector._data_input.send(second_msg)
    collector._data_input.send(next_group_msg)

    collector.run()
