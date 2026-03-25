from typing import List, Optional, Tuple, Union

import depthai as dai
import numpy as np

from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode


class Tiling(BaseThreadedHostNode):
    """Produces tiling ImageManipConfig groups and supports runtime reconfiguration.

    The node computes a :class:`dai.MessageGroup` of :class:`dai.ImageManipConfig`
    messages from the current tiling configuration. An internal Script node caches
    the latest config group from the ``cfg`` input using ``tryGet()`` and emits that
    group whenever a message arrives on the ``preview`` input.

    The main intended downstream consumer is :class:`depthai_nodes.node.FrameCropper`
    configured via ``fromManipConfigs``.
    """

    SCRIPT_CONTENT = """
try:
    def clone_message_group(msg_group):
        new_group = MessageGroup()
        for name in msg_group.getMessageNames():
            new_group[name] = msg_group[name]
        return new_group

    latest_cfg = node.inputs['cfg'].get()

    while True:
        preview = node.inputs['preview'].get()
        cfg = node.inputs['cfg'].tryGet()
        if cfg is not None:
            latest_cfg = cfg

        cfg_to_send = clone_message_group(latest_cfg)
        cfg_to_send.setSequenceNum(preview.getSequenceNum())
        cfg_to_send.setTimestamp(preview.getTimestamp())
        cfg_to_send.setTimestampDevice(preview.getTimestampDevice())
        node.outputs['cfg_group'].send(cfg_to_send)

except Exception as e:
    node.error(str(e))
"""

    def __init__(self) -> None:
        super().__init__()

        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(self.SCRIPT_CONTENT)
        self._script.outputs["cfg_group"].setPossibleDatatypes(
            [(dai.DatatypeEnum.MessageGroup, True)]
        )

        self._cfg_out = self.createOutput()
        self._crop_configs: List[dai.ImageManipConfig] = []
        self._cfg_group: Optional[dai.MessageGroup] = None
        self._is_built = False
        self._logger.debug("Tiling initialized")

    @property
    def out(self) -> dai.Node.Output:
        return self._script.outputs["cfg_group"]

    @property
    def tile_count(self) -> int:
        return len(self._crop_configs)

    def build(
        self,
        overlap: float,
        preview: dai.Node.Output,
        grid_size: Tuple[int, int],
        img_shape: Tuple[int, int],
        nn_shape: Tuple[int, int],
        resize_mode: dai.ImageManipConfig.ResizeMode,
        global_detection: bool = False,
        grid_matrix: Union[np.ndarray, List, None] = None,
    ) -> "TilingNew":
        self.setTilingConfig(
            overlap=overlap,
            grid_size=grid_size,
            img_shape=img_shape,
            nn_shape=nn_shape,
            resize_mode=resize_mode,
            global_detection=global_detection,
            grid_matrix=grid_matrix,
        )

        self._cfg_out.link(self._script.inputs["cfg"])
        preview.link(self._script.inputs["preview"])
        self._is_built = True

        self._logger.debug(
            "Tiling built with overlap=%s, grid_size=%s, img_shape=%s, nn_shape=%s, global_detection=%s",
            overlap,
            grid_size,
            img_shape,
            nn_shape,
            global_detection,
        )
        return self

    def setTilingConfig(
        self,
        overlap: float,
        grid_size: Tuple[int, int],
        img_shape: Tuple[int, int],
        nn_shape: Tuple[int, int],
        resize_mode: dai.ImageManipConfig.ResizeMode,
        global_detection: bool = False,
        grid_matrix: Union[np.ndarray, List, None] = None,
    ) -> None:
        tile_positions = self._computeTilePositions(
            overlap=overlap,
            grid_size=grid_size,
            img_shape=img_shape,
            grid_matrix=grid_matrix,
            global_detection=global_detection,
        )
        self._crop_configs = self._generateManipConfigs(
            tile_positions=tile_positions,
            nn_shape=nn_shape,
            resize_mode=resize_mode,
        )
        self._cfg_group = self._createMessageGroup(self._crop_configs)

    def run(self) -> None:
        while self._pipeline.isRunning():
            if self._cfg_group is not None:
                self._cfg_out.send(self._cfg_group)
                self._cfg_group = None
                self._pipeline.wait(0.1)

    def _createMessageGroup(
        self, crop_configs: List[dai.ImageManipConfig]
    ) -> dai.MessageGroup:
        msg_group = dai.MessageGroup()
        for i, cfg in enumerate(crop_configs):
            msg_group[str(i)] = cfg
        return msg_group

    def _computeTilePositions(
        self,
        overlap: float,
        grid_size: Tuple[int, int],
        img_shape: Tuple[int, int],
        grid_matrix: Union[np.ndarray, List, None],
        global_detection: bool,
    ) -> List[Tuple[int, int, int, int]]:
        tile_dims = self._calculateTiles(grid_size, img_shape, overlap)
        if grid_matrix is None:
            n_tiles_w, n_tiles_h = grid_size
            grid_matrix = [
                [j + i * n_tiles_w for j in range(n_tiles_w)] for i in range(n_tiles_h)
            ]
        if grid_size != (len(grid_matrix[0]), len(grid_matrix)):
            raise ValueError("Grid matrix dimensions do not match the grid size.")

        n_tiles_w, n_tiles_h = grid_size
        img_width, img_height = img_shape
        tile_width, tile_height = tile_dims

        labels = [[-1 for _ in range(n_tiles_w)] for _ in range(n_tiles_h)]
        component_id = 0

        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                if labels[i][j] != -1:
                    continue

                index_value = grid_matrix[i][j]
                queue = [(i, j)]
                while queue:
                    ci, cj = queue.pop()
                    if labels[ci][cj] != -1:
                        continue
                    if grid_matrix[ci][cj] != index_value:
                        continue

                    labels[ci][cj] = component_id

                    for ni, nj in [
                        (ci - 1, cj),
                        (ci + 1, cj),
                        (ci, cj - 1),
                        (ci, cj + 1),
                    ]:
                        if 0 <= ni < n_tiles_h and 0 <= nj < n_tiles_w:
                            if (
                                labels[ni][nj] == -1
                                and grid_matrix[ni][nj] == index_value
                            ):
                                queue.append((ni, nj))

                component_id += 1

        components = {}
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                comp_id = labels[i][j]
                if comp_id not in components:
                    components[comp_id] = []
                components[comp_id].append((i, j))

        tile_positions = []
        if global_detection:
            tile_positions.append((0, 0, img_width, img_height))

        for positions in components.values():
            x1_list = []
            y1_list = []
            x2_list = []
            y2_list = []

            for i, j in positions:
                x1_tile = int(j * tile_width * (1 - overlap))
                y1_tile = int(i * tile_height * (1 - overlap))
                x2_tile = min(int(x1_tile + tile_width), img_width)
                y2_tile = min(int(y1_tile + tile_height), img_height)

                x1_list.append(x1_tile)
                y1_list.append(y1_tile)
                x2_list.append(x2_tile)
                y2_list.append(y2_tile)

            tile_positions.append(
                (min(x1_list), min(y1_list), max(x2_list), max(y2_list))
            )

        return tile_positions

    def _calculateTiles(
        self, grid_size: Tuple[int, int], img_shape: Tuple[int, int], overlap: float
    ) -> np.ndarray:
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be in the range [0,1).")

        n_tiles_w, n_tiles_h = grid_size
        a = np.array(
            [
                [n_tiles_w * (1 - overlap) + overlap, 0],
                [0, n_tiles_h * (1 - overlap) + overlap],
            ]
        )
        b = np.array(img_shape)
        return np.linalg.inv(a).dot(b)

    def _generateManipConfigs(
        self,
        tile_positions: List[Tuple[int, int, int, int]],
        nn_shape: Tuple[int, int],
        resize_mode: dai.ImageManipConfig.ResizeMode,
    ) -> List[dai.ImageManipConfig]:
        return [
            self._getManipConfig(tile_info, nn_shape, resize_mode)
            for tile_info in tile_positions
        ]

    def _getManipConfig(
        self,
        tile_info: Tuple[int, int, int, int],
        nn_shape: Tuple[int, int],
        resize_mode: dai.ImageManipConfig.ResizeMode,
    ) -> dai.ImageManipConfig:
        x1, y1, x2, y2 = tile_info
        w = x2 - x1
        h = y2 - y1

        cfg = dai.ImageManipConfig()
        cfg.addCrop(x1, y1, w, h)
        cfg.setOutputSize(nn_shape[0], nn_shape[1], resize_mode)
        return cfg