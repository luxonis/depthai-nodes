from typing import List, Tuple, Union

import depthai as dai
import numpy as np

from depthai_nodes.logging import get_logger


class Tiling(dai.node.ThreadedHostNode):
    """Manages tiling of input frames for neural network processing, divides frames into
    overlapping tiles based on configuration parameters, and creates ImgFrames for each
    tile to be sent to a neural network node.

    @ivar overlap: Overlap between adjacent tiles, valid in [0,1).
    @type overlap: float
    @ivar grid_size: Grid size (number of tiles horizontally and vertically).
    @type grid_size: tuple
    @ivar grid_matrix: The matrix representing the grid of tiles.
    @type grid_matrix: list
    @ivar nn_shape: Shape of the neural network input.
    @type nn_shape: tuple
    @ivar x: Vector representing the tile's dimensions.
    @type x: list
    @ivar tile_positions: Coordinates and scaled sizes of the tiles.
    @type tile_positions: list
    @ivar img_shape: Shape of the original input image.
    @type img_shape: tuple
    @ivar global_detection: Whether to use global detection.
    @type global_detection: bool
    """

    SCRIPT_CONTENT = """
try:
    # Get initial configurations that will be sent
    # for every ImgFrame
    cfg_count_msg = node.inputs['cfg_count'].get()
    cfg_count = cfg_count_msg.getData()[0]
    configs = []
    for i in range(cfg_count):
        cfg = node.inputs['cfg'].get()
        configs.append(cfg)

    while True:
        frame = node.inputs['preview'].get()
        for cfg in configs:
            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)

except Exception as e:
    node.warn(str(e))
"""

    def __init__(self) -> None:
        """Initializes the Tiling node, setting default attributes like overlap, grid
        size, and tile positions."""
        super().__init__()
        self._logger = get_logger(self.__class__.__name__)

        self._pipeline = self.getParentPipeline()

        self.name = "Tiling"

        self.cropper_image_manip = self._pipeline.create(dai.node.ImageManip)
        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(self.SCRIPT_CONTENT)

        self._cfg_out = self.createOutput()
        self._cfg_count = self.createOutput()
        self._crop_configs: List[dai.ImageManipConfig] = []
        self._logger.debug("Tiling initialized")

    def build(
        self,
        overlap: float,
        img_output: dai.Node.Output,
        grid_size: Tuple,
        img_shape: Tuple,
        nn_shape: Tuple[int, int],
        resize_mode: dai.ImageManipConfig.ResizeMode,
        global_detection: bool = False,
        grid_matrix: Union[np.ndarray, List, None] = None,
    ) -> "Tiling":
        """Configures the Tiling node with grid size, overlap, image and neural network
        shapes, and other necessary parameters.

        @param overlap: Overlap between adjacent tiles, valid in [0,1).
        @type overlap: float
        @param img_output: The node from which the frames are sent.
        @type img_output: dai.Node.Output
        @param grid_size: Number of tiles horizontally and vertically.
        @type grid_size: tuple
        @param img_shape: Shape of the original image.
        @type img_shape: tuple
        @param nn_shape: Shape of the neural network input.
        @type nn_shape: tuple
        @param global_detection: Whether to perform global detection. Defaults to False.
        @type global_detection: bool
        @param grid_matrix: Predefined matrix for tiling. Defaults to None.
        @type grid_matrix: list or None
        @return: Returns self for method chaining.
        @rtype: Tiling
        """
        self._initCropConfigs(
            overlap=overlap,
            grid_size=grid_size,
            img_shape=img_shape,
            nn_shape=nn_shape,
            resize_mode=resize_mode,
            global_detection=global_detection,
            grid_matrix=grid_matrix,
        )

        self._cfg_out.link(self._script.inputs["cfg"])
        self._cfg_count.link(self._script.inputs["cfg_count"])
        img_output.link(self._script.inputs["preview"])
        self._script.outputs["manip_cfg"].link(self.cropper_image_manip.inputConfig)
        self._script.outputs["manip_img"].link(self.cropper_image_manip.inputImage)

        self._logger.debug(
            f"Tiling built with overlap={overlap}, grid_size={grid_size}, img_shape={img_shape}, nn_shape={nn_shape}, global_detection={global_detection}"
        )
        return self

    def run(self):
        """Send configuration to script node."""
        buff = dai.Buffer()
        buff.setData(np.array([self.tile_count], dtype=np.uint8))
        self._cfg_count.send(buff)

        for cfg in self._crop_configs:
            self._cfg_out.send(cfg)

    def _initCropConfigs(
        self,
        overlap: float,
        grid_size: Tuple,
        img_shape: Tuple,
        nn_shape: Tuple[int, int],
        resize_mode: dai.ImageManipConfig.ResizeMode,
        global_detection: bool = False,
        grid_matrix: Union[np.ndarray, List, None] = None,
    ):
        """Initializes the ImgManipConfig cropping configurations for the tiles."""
        tile_positions = self._computeTilePositions(
            overlap=overlap,
            grid_size=grid_size,
            img_shape=img_shape,
            grid_matrix=grid_matrix,
            global_detection=global_detection,
        )
        self._crop_configs = self._generateManipConfigs(
            tile_positions, nn_shape, resize_mode
        )

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

    def _calculateTiles(
        self, grid_size: Tuple, img_shape: Tuple, overlap: float
    ) -> np.ndarray:
        """Calculates the dimensions (x, y) of each tile given the grid size, image
        shape, and overlap.

        @param grid_size: The number of tiles in width and height.
        @type grid_size: tuple
        @param img_shape: The dimensions (width, height) of the input image.
        @type img_shape: tuple
        @param overlap: The overlap between adjacent tiles, valid in the range [0,1).
        @type overlap: float
        @return: The dimensions (width, height) of each tile.
        @rtype: np.ndarray
        """
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be in the range [0,1).")
        n_tiles_w, n_tiles_h = grid_size

        A = np.array(
            [
                [n_tiles_w * (1 - overlap) + overlap, 0],
                [0, n_tiles_h * (1 - overlap) + overlap],
            ]
        )

        b = np.array(img_shape)

        tile_dims = np.linalg.inv(A).dot(b)

        return tile_dims

    def _computeTilePositions(
        self,
        overlap: float,
        grid_size: Tuple,
        img_shape: Tuple,
        grid_matrix: Union[np.ndarray, List, None],
        global_detection: bool,
    ):
        """Computes and stores the tile positions and their scaled dimensions based on
        the grid matrix and overlap.

        This function is responsible for determining how the image is divided into tiles
        and how each tile maps back to the original image coordinates.
        """
        x = self._calculateTiles(grid_size, img_shape, overlap)
        if grid_matrix is None:
            n_tiles_w, n_tiles_h = grid_size
            grid_matrix = [
                [j + i * n_tiles_w for j in range(n_tiles_w)] for i in range(n_tiles_h)
            ]
        if grid_size != (len(grid_matrix[0]), len(grid_matrix)):
            raise ValueError("Grid matrix dimensions do not match the grid size.")

        n_tiles_w, n_tiles_h = grid_size
        img_width, img_height = img_shape

        tile_width, tile_height = x

        # labels to keep track of visited and unvisited tiles (-1 means unvisited)
        labels = [[-1 for _ in range(n_tiles_w)] for _ in range(n_tiles_h)]
        component_id = 0

        # Find connected components using a depth-first search
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                if labels[i][j] != -1:
                    # Already visited, skip
                    continue

                # Start a new component
                index_value = grid_matrix[i][j]
                queue = [(i, j)]
                while queue:
                    ci, cj = queue.pop()
                    if labels[ci][cj] != -1:
                        # Already visited, skip
                        continue
                    if grid_matrix[ci][cj] != index_value:
                        # Not part of the same component, skip
                        continue
                    # this tile is part of the current component, give it a label
                    labels[ci][cj] = component_id

                    # BFS: Check neighbors (up, down, left, right)
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
                                # this tile is part of the current component, add it to the queue to explore its neighbors
                                queue.append((ni, nj))

                # queue is empty, the current component is fully explored, move on to the next component
                component_id += 1

        # Group tiles by component
        components = {}
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                comp_id = labels[i][j]
                if comp_id not in components:
                    components[comp_id] = []
                components[comp_id].append((i, j))

        tile_positions = []
        # add a whole image as a tile with index 0 (hence the first to go)
        if global_detection:
            tile_positions.append((0, 0, img_width, img_height))

        # Compute the bounding box for each component
        for _, positions in components.items():
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

            # Compute the bounding box for the merged tile
            x1 = min(x1_list)
            y1 = min(y1_list)
            x2 = max(x2_list)
            y2 = max(y2_list)

            tile_positions.append((x1, y1, x2, y2))

        return tile_positions

    def _generateManipConfigs(
        self,
        tile_positions: List[Tuple[int, int, int, int]],
        nn_shape: Tuple[int, int],
        resize_mode: dai.ImageManipConfig.ResizeMode,
    ):
        """Creates ImageManipConfig from tile positions."""
        crop_configs = []
        for tile_info in tile_positions:
            cfg = self._getManipConfig(tile_info, nn_shape, resize_mode)
            crop_configs.append(cfg)
        return crop_configs

    @property
    def tile_count(self):
        return len(self._crop_configs)

    @property
    def out(self):
        return self.cropper_image_manip.out
