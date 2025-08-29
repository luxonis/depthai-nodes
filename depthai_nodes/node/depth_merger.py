from typing import Union

import depthai as dai

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.node.base_host_node import BaseHostNode

from .host_spatials_calc import HostSpatialsCalc


class DepthMerger(BaseHostNode):
    """DepthMerger is a custom host node for merging 2D detections with depth
    information to produce spatial detections.

    Attributes
    ----------
    output : dai.Node.Output
        The output of the DepthMerger node containing dai.SpatialImgDetections.
    shrinking_factor : float
        The shrinking factor for the bounding box. 0 means no shrinking. The factor means the percentage of the bounding box to shrink from each side.

    Usage
    -----
    depth_merger = pipeline.create(DepthMerger).build(
        output_2d=nn.out,
        output_depth=stereo.depth
    )
    """

    def __init__(self, shrinking_factor: float = 0) -> None:
        super().__init__()

        # TODO: We should make it consistant and use either output or out - IMO out is preferred to match DAI
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.SpatialImgDetections, True)
            ]
        )

        self.shrinking_factor = shrinking_factor
        self._logger.debug(
            f"DepthMerger initialized with shrinking_factor={shrinking_factor}"
        )

    def build(
        self,
        output_2d: dai.Node.Output,
        output_depth: dai.Node.Output,
        calib_data: dai.CalibrationHandler,
        depth_alignment_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
        shrinking_factor: float = 0,
    ) -> "DepthMerger":
        self.link_args(output_2d, output_depth)
        self.shrinking_factor = shrinking_factor
        self.host_spatials_calc = HostSpatialsCalc(calib_data, depth_alignment_socket)
        self._logger.debug(
            f"DepthMerger built with shrinking_factor={shrinking_factor}"
        )
        return self

    def process(self, message_2d: dai.Buffer, depth: dai.ImgFrame) -> None:
        self._logger.debug("Processing new input")
        spatial_dets = self._transform(message_2d, depth)
        self._logger.debug("Spatial detections message created")
        self.output.send(spatial_dets)  # type: ignore
        self._logger.debug("Message sent successfully")

    def _transform(
        self, message_2d: dai.Buffer, depth: dai.ImgFrame
    ) -> Union[dai.SpatialImgDetections, dai.SpatialImgDetection]:
        """Transforms 2D detections into spatial detections based on the depth frame."""
        if isinstance(message_2d, dai.ImgDetection):
            return self._detection_to_spatial(message_2d, depth)
        elif isinstance(message_2d, dai.ImgDetections):
            return self._detections_to_spatial(message_2d, depth)
        elif isinstance(message_2d, ImgDetectionExtended):
            return self._detection_to_spatial(message_2d, depth)
        elif isinstance(message_2d, ImgDetectionsExtended):
            return self._detections_to_spatial(message_2d, depth)
        else:
            raise ValueError(f"Unknown message type: {type(message_2d)}")

    def _detection_to_spatial(
        self,
        detection: Union[dai.ImgDetection, ImgDetectionExtended],
        depth: dai.ImgFrame,
    ) -> dai.SpatialImgDetection:
        """Converts a single 2D detection into a spatial detection using the depth
        frame."""
        depth_frame = depth.getCvFrame()
        x_len = depth_frame.shape[1]
        y_len = depth_frame.shape[0]
        xmin = (
            detection.rotated_rect.getOuterRect()[0]
            if isinstance(detection, ImgDetectionExtended)
            else detection.xmin
        )
        ymin = (
            detection.rotated_rect.getOuterRect()[1]
            if isinstance(detection, ImgDetectionExtended)
            else detection.ymin
        )
        xmax = (
            detection.rotated_rect.getOuterRect()[2]
            if isinstance(detection, ImgDetectionExtended)
            else detection.xmax
        )
        ymax = (
            detection.rotated_rect.getOuterRect()[3]
            if isinstance(detection, ImgDetectionExtended)
            else detection.ymax
        )
        xmin_corrected = xmin + (xmax - xmin) * self.shrinking_factor
        ymin_corrected = ymin + (ymax - ymin) * self.shrinking_factor
        xmax_corrected = xmax - (xmax - xmin) * self.shrinking_factor
        ymax_corrected = ymax - (ymax - ymin) * self.shrinking_factor
        roi = [
            self._get_index(xmin_corrected, x_len),
            self._get_index(ymin_corrected, y_len),
            self._get_index(xmax_corrected, x_len),
            self._get_index(ymax_corrected, y_len),
        ]
        spatials = self.host_spatials_calc.calc_spatials(depth, roi)

        spatial_img_detection = dai.SpatialImgDetection()
        spatial_img_detection.xmin = xmin
        spatial_img_detection.ymin = ymin
        spatial_img_detection.xmax = xmax
        spatial_img_detection.ymax = ymax
        spatial_img_detection.spatialCoordinates = dai.Point3f(
            spatials["x"], spatials["y"], spatials["z"]
        )

        spatial_img_detection.confidence = detection.confidence
        spatial_img_detection.label = 0 if detection.label == -1 else detection.label
        return spatial_img_detection

    def _detections_to_spatial(
        self,
        detections: Union[dai.ImgDetections, ImgDetectionsExtended],
        depth: dai.ImgFrame,
    ) -> dai.SpatialImgDetections:
        """Converts multiple 2D detections into spatial detections using the depth
        frame."""
        new_dets = dai.SpatialImgDetections()
        new_dets.detections = [
            self._detection_to_spatial(d, depth) for d in detections.detections
        ]
        new_dets.setSequenceNum(detections.getSequenceNum())
        new_dets.setTimestamp(detections.getTimestamp())
        new_dets.setTimestampDevice(detections.getTimestampDevice())
        transformation = detections.getTransformation()
        if transformation is not None:
            new_dets.setTransformation(transformation)
        return new_dets

    def _get_index(self, relative_coord: float, dimension_len: int) -> int:
        """Converts a relative coordinate to an absolute index within the given
        dimension length."""
        bounded_coord = min(1, relative_coord)
        return max(0, int(bounded_coord * dimension_len) - 1)
