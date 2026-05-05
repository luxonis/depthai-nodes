from unittest.mock import Mock

import depthai as dai

from depthai_nodes.node.tiling import Tiling


class OutputRecorder:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


class PipelineStub:
    def __init__(self):
        self._running = True

    def isRunning(self):
        running = self._running
        self._running = False
        return running

    def wait(self, _seconds):
        return None


def create_tiling_node() -> Tiling:
    node = object.__new__(Tiling)
    node._pipeline = PipelineStub()
    node._cfg_out = OutputRecorder()
    node._logger = Mock()
    return node.build(
        overlap=0.0,
        gridSize=(2, 1),
        canvasShape=(100, 50),
        resizeShape=(32, 32),
        resizeMode=dai.ImageManipConfig.ResizeMode.CENTER_CROP,
        globalDetection=True,
    )


def test_compute_tile_positions_with_global_detection():
    tiling = create_tiling_node()

    positions = tiling.tilePositions

    assert positions == [(0, 0, 100, 50), (0, 0, 50, 50), (50, 0, 100, 50)]


def test_create_message_group_contains_all_configs():
    tiling = create_tiling_node()
    configs = [dai.ImageManipConfig(), dai.ImageManipConfig()]

    msg_group = tiling._createMessageGroup(configs)

    assert isinstance(msg_group, dai.MessageGroup)
    assert msg_group.getNumMessages() == 2
    assert msg_group.getMessageNames() == ["0", "1"]
    assert msg_group["0"] is configs[0]
    assert msg_group["1"] is configs[1]


def test_update_tiling_config_sends_updated_message_group():
    tiling = create_tiling_node()

    tiling.updateTilingConfig(
        gridSize=(1, 1),
        resizeShape=(32, 32),
        resizeMode=dai.ImageManipConfig.ResizeMode.CENTER_CROP,
        globalDetection=False,
    )

    assert tiling.tileCount == 1
    assert len(tiling._cfg_out.sent) == 1
    assert isinstance(tiling._cfg_out.sent[0], dai.MessageGroup)
    assert tiling._cfg_out.sent[0].getNumMessages() == 1


def test_run_sends_message_group_for_current_tiling_config():
    tiling = create_tiling_node()

    tiling.run()

    assert len(tiling._cfg_out.sent) == 1
    assert isinstance(tiling._cfg_out.sent[0], dai.MessageGroup)
    assert tiling._cfg_out.sent[0].getNumMessages() == 3
