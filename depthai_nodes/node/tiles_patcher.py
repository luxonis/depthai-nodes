import datetime
from typing import List, Optional, Tuple, Type, Union

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.img_detections import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
)
from depthai_nodes.message.keypoints import Keypoints
from depthai_nodes.node.base_host_node import BaseHostNode
from depthai_nodes.node.utils import nms_detections

from .tiling import Tiling


class TilesPatcher(BaseHostNode):
    """Handles the processing of tiled frames from neural network (NN) outputs, maps the
    detections from tiles back into the global frame, and sends out the combined
    detections for further processing.

    @ivar conf_thresh: Confidence threshold for filtering detections.
    @type conf_thresh: float
    @ivar iou_thresh: IOU threshold for non-max suppression.
    @type iou_thresh: float
    @ivar tile_manager: Manager responsible for handling tiling configurations.
    @type tile_manager: Tiling
    @ivar tile_buffer: Buffer to store tile detections temporarily.
    @type tile_buffer: list
    @ivar current_timestamp: Timestamp for the current frame being processed.
    @type current_timestamp: float
    @ivar expected_tiles_count: Number of tiles expected per frame.
    @type expected_tiles_count: int
    """

    def __init__(self) -> None:
        """Initializes the TilesPatcher node, sets default thresholds for confidence and
        IOU, and initializes buffers for tile processing."""
        super().__init__()
        self.name = "TilesPatcher"
        self.tile_manager = None
        self.conf_thresh = 0.3
        self.iou_thresh = 0.4

        self.tile_buffer = []
        self.segmentation_buffer = []
        self.current_timestamp = None
        self.expected_tiles_count = 0
        self._input_class = dai.ImgDetections
        self._logger.debug("TilesPatcher initialized")

    def build(
        self, tile_manager: Tiling, nn: dai.Node.Output, conf_thresh=0.3, iou_thresh=0.4
    ) -> "TilesPatcher":
        """Configures the TilesPatcher node with the tile manager and links the neural
        network's output.

        @param tile_manager: The tiling manager responsible for tile positions and
            dimensions.
        @type tile_manager: Tiling
        @param nn: The output of the neural network node from which detections are
            received.
        @type nn: dai.Node.Output
        @param conf_thresh: Confidence threshold for filtering detections (default:
            0.3).
        @type conf_thresh: float
        @param iou_thresh: IOU threshold for non-max suppression (default: 0.4).
        @type iou_thresh: float
        @return: Returns self for method chaining.
        @rtype: TilesPatcher
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.tile_manager = tile_manager
        if (
            self.tile_manager.x is None
            or self.tile_manager.grid_size is None
            or self.tile_manager.overlap is None
        ):
            raise ValueError("Tile dimensions, grid size, or overlap not initialized.")
        self.expected_tiles_count = len(self.tile_manager.tile_positions)
        self.sendProcessingToPipeline(True)
        self.link_args(nn)
        self._logger.debug(
            f"TilesPatcher built with conf_thresh={conf_thresh}, iou_thresh={iou_thresh}"
        )
        return self

    def process(self, nn_output: dai.Buffer) -> None:
        """Processes each neural network output (detections) by mapping them from tiled
        patches back into the global frame and buffering them until all tiles for the
        current frame are processed.

        @param nn_output: The detections from the neural network's output.
        @type nn_output: dai.ImgDetections
        """
        assert isinstance(
            nn_output, (dai.ImgDetections, ImgDetectionsExtended)
        ), "Invalid input type"

        self._logger.debug("Processing new input")
        timestamp = nn_output.getTimestamp()
        device_timestamp = nn_output.getTimestampDevice()
        sequence_num = nn_output.getSequenceNum()
        transformation = nn_output.getTransformation()

        if self.current_timestamp is None:
            self.current_timestamp = timestamp

        if self.current_timestamp != timestamp and len(self.tile_buffer) > 0:
            # new frame started, send the output for the previous frame
            self._send_output(
                self.current_timestamp,
                device_timestamp,
                transformation,
                sequence_num,
                type(nn_output),
            )
            self.tile_buffer = []
            self.segmentation_buffer = []

        self.current_timestamp = timestamp

        if isinstance(nn_output, ImgDetectionsExtended):
            self.segmentation_buffer.append((nn_output.masks, sequence_num))
        bboxes: Union[
            List[dai.ImgDetection], List[ImgDetectionExtended]
        ] = nn_output.detections
        mapped_bboxes = self._map_bboxes_to_global_frame(bboxes, sequence_num)
        self.tile_buffer.append(mapped_bboxes)

        if len(self.tile_buffer) == self.expected_tiles_count:
            self._send_output(
                timestamp,
                device_timestamp,
                transformation,
                sequence_num,
                type(nn_output),
            )
            self.tile_buffer = []
            self.segmentation_buffer = []

    def _map_img_detection_to_global_frame(
        self, detection: dai.ImgDetection, tile_info: dict, nn_shape: Tuple[int, int]
    ):
        nn_width, nn_height = nn_shape

        tile_x1, tile_y1, tile_x2, tile_y2 = tile_info["coords"]
        tile_actual_width = tile_x2 - tile_x1
        tile_actual_height = tile_y2 - tile_y1

        # Scaled dimensions (after resizing to fit NN input)
        scaled_width, scaled_height = tile_info["scaled_size"]

        # Offsets due to padding
        x_offset = (nn_width - scaled_width) // 2
        y_offset = (nn_height - scaled_height) // 2

        # Scaling factors from scaled tile back to original tile dimensions
        scale_x = tile_actual_width / scaled_width
        scale_y = tile_actual_height / scaled_height

        # Ensure xmin < xmax and ymin < ymax
        xmin = min(detection.xmin, detection.xmax)
        ymin = min(detection.ymin, detection.ymax)
        xmax = max(detection.xmin, detection.xmax)
        ymax = max(detection.ymin, detection.ymax)

        # Convert bbox coordinates from normalized to NN input dimensions
        bbox_xmin_nn = xmin * nn_width
        bbox_ymin_nn = ymin * nn_height
        bbox_xmax_nn = xmax * nn_width
        bbox_ymax_nn = ymax * nn_height

        # Adjust for padding offsets to get coordinates in the scaled tile
        bbox_xmin_scaled = bbox_xmin_nn - x_offset
        bbox_ymin_scaled = bbox_ymin_nn - y_offset
        bbox_xmax_scaled = bbox_xmax_nn - x_offset
        bbox_ymax_scaled = bbox_ymax_nn - y_offset

        # Ensure coordinates are within the scaled tile dimensions
        bbox_xmin_scaled = max(0, min(bbox_xmin_scaled, scaled_width))
        bbox_ymin_scaled = max(0, min(bbox_ymin_scaled, scaled_height))
        bbox_xmax_scaled = max(0, min(bbox_xmax_scaled, scaled_width))
        bbox_ymax_scaled = max(0, min(bbox_ymax_scaled, scaled_height))

        # Map to original tile coordinates
        bbox_xmin_tile = bbox_xmin_scaled * scale_x
        bbox_ymin_tile = bbox_ymin_scaled * scale_y
        bbox_xmax_tile = bbox_xmax_scaled * scale_x
        bbox_ymax_tile = bbox_ymax_scaled * scale_y

        # Map to global image coordinates
        x1_global = tile_x1 + bbox_xmin_tile
        y1_global = tile_y1 + bbox_ymin_tile
        x2_global = tile_x1 + bbox_xmax_tile
        y2_global = tile_y1 + bbox_ymax_tile

        # Normalize global coordinates
        img_width, img_height = self.tile_manager.img_shape  # type: ignore
        normalized_detection = dai.ImgDetection()
        normalized_detection.label = detection.label
        normalized_detection.confidence = detection.confidence
        normalized_detection.xmin = x1_global / img_width
        normalized_detection.ymin = y1_global / img_height
        normalized_detection.xmax = x2_global / img_width
        normalized_detection.ymax = y2_global / img_height

        return normalized_detection

    def _map_img_detection_extended_to_global_frame(
        self,
        detection: ImgDetectionExtended,
        tile_info: dict,
        nn_shape: Tuple[int, int],
    ):
        points = detection.rotated_rect.getPoints()
        mapped_corner_pts = [
            self._map_point_to_global_frame((pt.x, pt.y), tile_info, nn_shape)
            for pt in points
        ]
        (center_x, center_y), (width, height), angle = cv2.minAreaRect(
            np.array(mapped_corner_pts, dtype=np.float32)
        )
        normalized_detection = detection.copy()
        normalized_detection.rotated_rect = (center_x, center_y, width, height, angle)
        kpts = Keypoints()
        mapped_kpts = []
        for kpt in detection.keypoints:
            coords = self._map_point_to_global_frame(
                (kpt.x, kpt.y), tile_info, nn_shape
            )
            new_kpt = kpt.copy()
            new_kpt.x = coords[0]
            new_kpt.y = coords[1]
            mapped_kpts.append(new_kpt)

        kpts.keypoints = mapped_kpts
        normalized_detection.keypoints = kpts
        return normalized_detection

    def _map_point_to_global_frame(
        self,
        point: Tuple[float, float],
        tile_info: dict,
        nn_shape: Tuple[int, int],
    ):
        nn_width, nn_height = nn_shape

        tile_x1, tile_y1, tile_x2, tile_y2 = tile_info["coords"]
        tile_actual_width = tile_x2 - tile_x1
        tile_actual_height = tile_y2 - tile_y1

        # Scaled dimensions (after resizing to fit NN input)
        scaled_width, scaled_height = tile_info["scaled_size"]

        # Offsets due to padding
        x_offset = (nn_width - scaled_width) // 2
        y_offset = (nn_height - scaled_height) // 2

        # Scaling factors from scaled tile back to original tile dimensions
        scale_x = tile_actual_width / scaled_width
        scale_y = tile_actual_height / scaled_height

        # Convert point coordinates from normalized to NN input dimensions
        point_nn = (point[0] * nn_width, point[1] * nn_height)

        # Adjust for padding offsets to get coordinates in the scaled tile
        point_scaled = (point_nn[0] - x_offset, point_nn[1] - y_offset)

        # Ensure coordinates are within the scaled tile dimensions
        point_scaled = (
            max(0, min(point_scaled[0], scaled_width)),
            max(0, min(point_scaled[1], scaled_height)),
        )

        # Map to original tile coordinates
        point_tile = (point_scaled[0] * scale_x, point_scaled[1] * scale_y)

        # Map to global image coordinates
        point_global = (tile_x1 + point_tile[0], tile_y1 + point_tile[1])

        # Normalize global coordinates
        img_width, img_height = self.tile_manager.img_shape  # type: ignore
        point_global_normalized = (
            point_global[0] / img_width,
            point_global[1] / img_height,
        )

        return point_global_normalized

    def _map_bboxes_to_global_frame(
        self,
        bboxes: Union[List[dai.ImgDetection], List[ImgDetectionExtended]],
        tile_index: int,
    ) -> Union[List[dai.ImgDetection], List[ImgDetectionExtended]]:
        """Maps bounding boxes from their local tile coordinates back to the global
        frame of the full image.

        @param bboxes: The bounding boxes to be mapped.
        @type bboxes: list[dai.ImgDetection]
        @param tile_index: The index of the tile being processed.
        @type tile_index: int
        @return: Mapped bounding boxes in the global image frame.
        @rtype: list[dai.ImgDetection]
        """
        tile_info = self._get_tile_info(tile_index)
        if (
            self.tile_manager is None
            or self.tile_manager.nn_shape is None
            or tile_info is None
        ):
            return []
        if tile_info is None:
            return []

        global_bboxes = []
        for bbox in bboxes:
            if isinstance(bbox, dai.ImgDetection):
                normalized_bbox = self._map_img_detection_to_global_frame(
                    bbox, tile_info, self.tile_manager.nn_shape
                )
            elif isinstance(bbox, ImgDetectionExtended):
                normalized_bbox = self._map_img_detection_extended_to_global_frame(
                    bbox, tile_info, self.tile_manager.nn_shape
                )
            else:
                raise TypeError(
                    f"Expected dai.ImgDetection or ImgDetectionExtended, got {type(bbox)}"
                )

            global_bboxes.append(normalized_bbox)

        return global_bboxes

    def _stitch_segmentation_maps(
        self,
        segmentation_maps: List[Tuple[np.ndarray, int]],
    ) -> np.ndarray:
        """Stitches segmentation maps from tiles back into the global frame and returns
        the full segmentation map.

        @param segmentation_maps: List of segmentation maps and their corresponding tile
            indices.
        @type segmentation_maps: list[tuple[np.ndarray, int]]
        @return: Stitched segmentation map.
        @rtype: np.ndarray
        """
        if self.tile_manager is None or self.tile_manager.nn_shape is None:
            raise RuntimeError("Tile manager or tile positions not initialized.")
        full_seg_map = np.zeros(
            (self.tile_manager.img_shape[1], self.tile_manager.img_shape[0]),  # type: ignore
            np.int16,
        )
        if len(segmentation_maps) == 0:
            return np.empty(0, dtype=np.int16)
        if all([seg_map.size == 0 for seg_map, _ in segmentation_maps]):
            return np.empty(0, dtype=np.int16)
        for seg_map, tile_index in segmentation_maps:
            tile_info = self._get_tile_info(tile_index)
            if tile_info is None:
                raise ValueError("Tile information not found.")

            nn_width, nn_height = self.tile_manager.nn_shape

            tile_x1, tile_y1, tile_x2, tile_y2 = tile_info["coords"]
            tile_actual_width = tile_x2 - tile_x1
            tile_actual_height = tile_y2 - tile_y1

            # Scaled dimensions (after resizing to fit NN input)
            scaled_width, scaled_height = tile_info["scaled_size"]

            # Offsets due to padding
            x_offset = (nn_width - scaled_width) // 2
            y_offset = (nn_height - scaled_height) // 2

            # Create empty segmentation map if it doesn't exist
            if seg_map.size == 0:
                seg_map = np.zeros((nn_width, nn_height), dtype=np.int16)

            # Remove padding from segmentation map
            unpadded_seg_map = seg_map[
                y_offset : y_offset + scaled_height, x_offset : x_offset + scaled_width
            ]
            resized_seg_map = cv2.resize(
                unpadded_seg_map,
                (tile_actual_width, tile_actual_height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Use this mask to ensure zero values do not overwrite non-zero values
            # in the full segmentation map
            non_zero_seg_map = resized_seg_map != 0
            full_seg_map[tile_y1:tile_y2, tile_x1:tile_x2][
                non_zero_seg_map
            ] = resized_seg_map[non_zero_seg_map]
        return full_seg_map

    def _get_tile_info(self, tile_index: int):
        """Retrieves the tile's coordinates and scaled dimensions based on the tile
        index.

        @param tile_index: The index of the tile.
        @type tile_index: int
        @return: Tile information for the given index.
        """
        if self.tile_manager is None or self.tile_manager.tile_positions is None:
            raise ValueError("Tile manager or tile positions not initialized.")
        if tile_index >= len(self.tile_manager.tile_positions):
            return None
        return self.tile_manager.tile_positions[tile_index]

    def _send_output(
        self,
        timestamp: datetime.timedelta,
        device_timestamp: datetime.timedelta,
        transformation: Optional[dai.ImgTransformation],
        sequence_num: int,
        input_type: Type[Union[dai.ImgDetections, ImgDetectionsExtended]],
    ) -> None:
        """Send the final combined bounding boxes as output when all tiles for a frame
        are processed.

        @param timestamp: The timestamp of the frame.
        @param device_timestamp: The timestamp of the frame on the device.
        """
        combined_bboxes: Union[List[dai.ImgDetection], List[ImgDetectionExtended]] = []
        for bboxes in self.tile_buffer:
            combined_bboxes.extend(bboxes)

        detection_list = nms_detections(
            combined_bboxes,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
        )
        if input_type == dai.ImgDetections:
            detections = dai.ImgDetections()
            detections.detections = detection_list
        elif input_type == ImgDetectionsExtended:
            detections = ImgDetectionsExtended()
            detections.masks = self._stitch_segmentation_maps(self.segmentation_buffer)
            detections.detections = detection_list
        else:
            raise ValueError("Unsupported input type")

        detections.setTimestamp(timestamp)
        detections.setTimestampDevice(device_timestamp)
        detections.setSequenceNum(sequence_num)
        if transformation is not None:
            detections.setTransformation(transformation)

        self._logger.debug("Detections message created")

        self.out.send(detections)

        self._logger.debug("Message sent successfully")
