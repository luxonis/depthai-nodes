import cv2
import depthai as dai
import numpy as np

from .utils.to_planar import to_planar


class Tiling(dai.node.HostNode):
    """Manages tiling of input frames for neural network processing, divides frames into
    overlapping tiles based on configuration parameters, and creates ImgFrames for each
    tile to be sent to a neural network node.

    Attributes:
        overlap (float): Overlap between adjacent tiles, valid in [0,1).
        grid_size (tuple): Grid size (number of tiles horizontally and vertically).
        grid_matrix (list): The matrix representing the grid of tiles.
        nn_shape (tuple): Shape of the neural network input.
        x (list): Vector representing the tile's dimensions.
        tile_positions (list): Coordinates and scaled sizes of the tiles.
        img_shape (tuple): Shape of the original input image.
        global_detection (bool): Whether to use global detection.
    """

    def __init__(self) -> None:
        """Initializes the Tiling node, setting default attributes like overlap, grid
        size, and tile positions."""
        super().__init__()
        self.name = "Tiling"
        self.overlap = None
        self.grid_size = None
        self.grid_matrix = None
        self.nn_shape = None
        self.x = None  # vector [x,y] of the tile's dimensions
        self.tile_positions = []
        self.img_shape = None
        self.global_detection = False

    def build(
        self,
        overlap: float,
        img_output: dai.Node.Output,
        grid_size: tuple,
        img_shape: tuple,
        nn_shape: tuple,
        global_detection: bool = False,
        grid_matrix=None,
    ) -> "Tiling":
        """Configures the Tiling node with grid size, overlap, image and neural network
        shapes, and other necessary parameters.

        Args:
            overlap (float): Overlap between adjacent tiles, valid in [0,1).
            img_output (dai.Node.Output): The node from which the frames are sent.
            grid_size (tuple): Number of tiles horizontally and vertically.
            img_shape (tuple): Shape of the original image.
            nn_shape (tuple): Shape of the neural network input.
            global_detection (bool, optional): Whether to perform global detection.
            grid_matrix (list, optional): Predefined matrix for tiling.

        Returns:
            Tiling: Returns self for method chaining.
        """
        self.sendProcessingToPipeline(True)
        self.link_args(img_output)
        self.overlap = overlap
        self.grid_size = grid_size
        self.grid_matrix = grid_matrix
        self.nn_shape = nn_shape
        self.img_shape = img_shape
        self.global_detection = global_detection
        self.x = self._calculate_tiles(grid_size, img_shape, overlap)
        self._compute_tile_positions()

        return self

    def process(self, img_frame) -> None:
        """Processes the input frame by cropping and tiling it, and sending each tile to
        the neural network input.

        Args:
            img_frame (dai.ImgFrame): The frame to be sent to a neural network.
        """
        frame: np.ndarray = img_frame.getCvFrame()

        if self.grid_size is None or self.x is None or self.nn_shape is None:
            raise ValueError("Grid size or tile dimensions are not initialized.")
        if self.tile_positions is None:
            raise ValueError("Tile positions are not initialized.")

        for index, tile_info in enumerate(self.tile_positions):
            x1, y1, x2, y2 = tile_info["coords"]
            scaled_width, scaled_height = tile_info["scaled_size"]
            tile = frame[y1:y2, x1:x2]
            tile_img_frame = self._create_img_frame(
                tile, img_frame, index, scaled_width, scaled_height
            )
            self.out.send(tile_img_frame)

    def _crop_to_nn_shape(
        self, frame: np.ndarray, scaled_width, scaled_height
    ) -> np.ndarray:
        """Crops and resizes the input tile to fit the neural network's input shape.
        Adds padding if necessary to preserve aspect ratio.

        Args:
            frame (np.ndarray): The tile to be cropped and resized.
            scaled_width (int): The width of the scaled tile.
            scaled_height (int): The height of the scaled tile.

        Returns:
            np.ndarray: The padded and resized tile.
        """
        if self.nn_shape is None:
            raise ValueError("NN shape is not initialized.")
        frame_resized = cv2.resize(
            frame, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST
        )
        frame_padded = np.zeros((self.nn_shape[1], self.nn_shape[0], 3), dtype=np.uint8)

        x_offset = (self.nn_shape[0] - scaled_width) // 2
        y_offset = (self.nn_shape[1] - scaled_height) // 2
        frame_padded[
            y_offset : y_offset + scaled_height, x_offset : x_offset + scaled_width
        ] = frame_resized

        return frame_padded

    def _create_img_frame(
        self, tile: np.ndarray, frame, tile_index, scaled_width, scaled_height
    ) -> dai.ImgFrame:
        """Creates an ImgFrame from the cropped tile and prepares it for input into the
        neural network. This ImgFrame contains the tiled image in a planar format ready
        for inference.

        Args:
            tile (np.ndarray): The cropped tile.
            frame: The original ImgFrame object, providing metadata such as timestamps.
            tile_index (int): Index of the current tile.
            scaled_width (int): Width of the scaled tile.
            scaled_height (int): Height of the scaled tile.

        Returns:
            dai.ImgFrame: The ImgFrame object with the tile data, ready for neural network inference.
        """
        if self.nn_shape is None:
            raise ValueError("NN shape is not initialized.")
        tile_padded = self._crop_to_nn_shape(tile, scaled_width, scaled_height)

        planar_tile = to_planar(tile_padded, self.nn_shape)

        img_frame = dai.ImgFrame()
        img_frame.setData(planar_tile)
        img_frame.setWidth(self.nn_shape[0])
        img_frame.setHeight(self.nn_shape[1])
        img_frame.setType(dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(frame.getTimestamp())
        img_frame.setTimestampDevice(frame.getTimestampDevice())
        img_frame.setInstanceNum(frame.getInstanceNum())
        img_frame.setSequenceNum(tile_index)

        return img_frame

    def _calculate_tiles(self, grid_size, img_shape, overlap):
        """Calculates the dimensions (x, y) of each tile given the grid size, image
        shape, and overlap.

        Args:
            grid_size (tuple): The number of tiles in width and height.
            img_shape (tuple): The dimensions (width, height) of the input image.
            overlap (float): The overlap between adjacent tiles, valid in the range [0,1).

        Returns:
            np.ndarray: The dimensions (width, height) of each tile.
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

    def _compute_tile_positions(self):
        """Computes and stores the tile positions and their scaled dimensions based on
        the grid matrix and overlap.

        This function is responsible for determining how the image is divided into tiles
        and how each tile maps back to the original image coordinates.
        """
        if (
            self.grid_size is None
            or self.overlap is None
            or self.x is None
            or self.img_shape is None
            or self.nn_shape is None
        ):
            raise ValueError(
                "Grid size, overlap, tile dimensions, or image shape not initialized."
            )
        if self.grid_matrix is None:
            n_tiles_w, n_tiles_h = self.grid_size
            self.grid_matrix = [
                [j + i * n_tiles_w for j in range(n_tiles_w)] for i in range(n_tiles_h)
            ]
        if self.grid_size != (len(self.grid_matrix[0]), len(self.grid_matrix)):
            raise ValueError("Grid matrix dimensions do not match the grid size.")

        n_tiles_w, n_tiles_h = self.grid_size
        img_width, img_height = self.img_shape

        tile_width, tile_height = self.x

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
                index_value = self.grid_matrix[i][j]
                queue = [(i, j)]
                while queue:
                    ci, cj = queue.pop()
                    if labels[ci][cj] != -1:
                        # Already visited, skip
                        continue
                    if self.grid_matrix[ci][cj] != index_value:
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
                                and self.grid_matrix[ni][nj] == index_value
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

        self.tile_positions = []
        # add a whole image as a tile with index 0 (hence the first to go)
        if self.global_detection:
            scale = min(self.nn_shape / img_width, self.nn_shape / img_height)
            scaled_width = int(img_width * scale)
            scaled_height = int(img_height * scale)
            self.tile_positions.append(
                {
                    "coords": (0, 0, img_width, img_height),
                    "scaled_size": (scaled_width, scaled_height),
                }
            )

        # Compute the bounding box for each component
        for _, positions in components.items():
            x1_list = []
            y1_list = []
            x2_list = []
            y2_list = []

            for i, j in positions:
                x1_tile = int(j * tile_width * (1 - self.overlap))
                y1_tile = int(i * tile_height * (1 - self.overlap))
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

            tile_actual_width = x2 - x1
            tile_actual_height = y2 - y1

            # the scaled dimenstion after being resized to fit nn_shape (precomputed and saved for optimization)
            scale_width = self.nn_shape[0] / tile_actual_width
            scale_height = self.nn_shape[1] / tile_actual_height
            scale = min(scale_width, scale_height)

            scaled_width = int(tile_actual_width * scale)
            scaled_height = int(tile_actual_height * scale)

            self.tile_positions.append(
                {
                    "coords": (x1, y1, x2, y2),
                    "scaled_size": (scaled_width, scaled_height),
                }
            )
