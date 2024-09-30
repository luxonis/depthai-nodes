import depthai as dai
from .utils.nms import nms_detections
from ..preprocessing.tiling import Tiling

class TilesPatcher(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Patcher"
        self.tile_manager = None
        self.conf_thresh = 0.3
        self.iou_thresh = 0.4

        self.tile_buffer = []
        self.current_timestamp = None
        self.expected_tiles_count = 0

    def set_conf_thresh(self, conf_thresh: float) -> None:
        self.conf_thresh = conf_thresh

    def set_iou_thresh(self, iou_thresh: float) -> None:
        self.iou_thresh = iou_thresh

    def build(self, tile_manager: Tiling, nn: dai.Node.Output):
        self.tile_manager = tile_manager
        if self.tile_manager.x is None or self.tile_manager.grid_size is None or self.tile_manager.overlap is None:
            raise ValueError("Tile dimensions, grid size, or overlap not initialized.")
        self.expected_tiles_count = len(self.tile_manager.tile_positions) 
        self.sendProcessingToPipeline(True)
        self.link_args(nn)
        return self

    def process(self, nn_output: dai.ImgDetections) -> None:
        timestamp = nn_output.getTimestamp()
        device_timestamp = nn_output.getTimestampDevice()

        if self.current_timestamp is None:
            self.current_timestamp = timestamp

        if self.current_timestamp != timestamp and len(self.tile_buffer) > 0:
            # new frame started, send the output for the previous frame
            self._send_output(self.current_timestamp, device_timestamp)
            self.tile_buffer = []

        self.current_timestamp = timestamp
        tile_index = nn_output.getSequenceNum()

        bboxes: list[dai.ImgDetection] = nn_output.detections 
        mapped_bboxes = self._map_bboxes_to_global_frame(bboxes, tile_index)
        self.tile_buffer.append(mapped_bboxes)

        if len(self.tile_buffer) == self.expected_tiles_count:
            self._send_output(timestamp, device_timestamp)
            self.tile_buffer = []

    def _map_bboxes_to_global_frame(self, bboxes: list[dai.ImgDetection], tile_index: int):
        tile_info = self._get_tile_info(tile_index)
        if self.tile_manager is None or self.tile_manager.nn_shape is None or tile_info is None:
            return []
        if tile_info is None:
            return []

        # Original tile coordinates in the global frame
        tile_x1, tile_y1, tile_x2, tile_y2 = tile_info['coords']
        tile_actual_width = tile_x2 - tile_x1
        tile_actual_height = tile_y2 - tile_y1

        # Scaled dimensions (after resizing to fit NN input)
        scaled_width, scaled_height = tile_info['scaled_size']
        nn_width, nn_height = self.tile_manager.nn_shape

        # Offsets due to padding
        x_offset = (nn_width - scaled_width) // 2
        y_offset = (nn_height - scaled_height) // 2

        # Scaling factors from scaled tile back to original tile dimensions
        scale_x = tile_actual_width / scaled_width
        scale_y = tile_actual_height / scaled_height

        global_bboxes = []
        for bbox in bboxes:
            # Convert bbox coordinates from normalized to NN input dimensions
            bbox_xmin_nn = bbox.xmin * nn_width
            bbox_ymin_nn = bbox.ymin * nn_height
            bbox_xmax_nn = bbox.xmax * nn_width
            bbox_ymax_nn = bbox.ymax * nn_height

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
            img_width, img_height = self.tile_manager.img_shape
            normalized_bbox = dai.ImgDetection()
            normalized_bbox.label = bbox.label
            normalized_bbox.confidence = bbox.confidence
            normalized_bbox.xmin = x1_global / img_width
            normalized_bbox.ymin = y1_global / img_height
            normalized_bbox.xmax = x2_global / img_width
            normalized_bbox.ymax = y2_global / img_height

            global_bboxes.append(normalized_bbox)

        return global_bboxes

    def _get_tile_info(self, tile_index: int):
        """
        Retrieves the tile's coordinates and scaled dimensions based on the tile index.
        """
        if self.tile_manager is None or self.tile_manager.tile_positions is None:
            raise ValueError("Tile manager or tile positions not initialized.")
        if tile_index >= len(self.tile_manager.tile_positions):
            return None
        return self.tile_manager.tile_positions[tile_index]

    def _send_output(self, timestamp, device_timestamp):
        """
        Send the final combined bounding boxes as output when all tiles for a frame are processed.
        """
        combined_bboxes: list[dai.ImgDetection] = []
        for bboxes in self.tile_buffer:
            combined_bboxes.extend(bboxes)

        if combined_bboxes:
            detection_list = nms_detections(
                combined_bboxes,
                conf_thresh=self.conf_thresh,
                iou_thresh=self.iou_thresh
            )
        else:
            detection_list = []

        # Create ImgDetections message
        detections = dai.ImgDetections()
        detections.setTimestamp(timestamp)
        detections.setTimestampDevice(device_timestamp)
        detections.detections = detection_list

        self.out.send(detections)
