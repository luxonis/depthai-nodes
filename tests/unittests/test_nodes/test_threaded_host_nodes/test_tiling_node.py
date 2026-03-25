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
    node._crop_configs = []
    node._cfg_group = None
    return node


def test_compute_tile_positions_with_global_detection():
    tiling = create_tiling_node()

    positions = tiling._computeTilePositions(
        overlap=0.0,
        grid_size=(2, 1),
        canvas_shape=(100, 50),
        grid_matrix=None,
        global_detection=True,
    )

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


def test_set_tiling_config_sends_updated_message_group():
    tiling = create_tiling_node()

    tiling.setTilingConfig(
        overlap=0.0,
        gridSize=(2, 1),
        canvasShape=(100, 50),
        resizeShape=(32, 32),
        resizeMode=dai.ImageManipConfig.ResizeMode.CENTER_CROP,
    )

    assert tiling.tile_count == 2
    assert isinstance(tiling._cfg_group, dai.MessageGroup)
    assert tiling._cfg_group.getNumMessages() == 2
    assert tiling._cfg_out.sent == []


def test_run_sends_initial_message_group():
    tiling = create_tiling_node()
    initial_group = dai.MessageGroup()
    tiling._cfg_group = initial_group

    tiling.run()

    assert tiling._cfg_out.sent == [initial_group]
    assert tiling._cfg_group is None
