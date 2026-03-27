from typing import Union

import depthai as dai

from depthai_nodes.node.base_host_node import BaseHostNode

from .host_spatials_calc import HostSpatialsCalc


class DepthMerger(BaseHostNode):
    """DepthMerger is a custom host node for merging 2D detections with depth
    information to produce spatial detections.

    Attributes
    ----------
    output : dai.Node.Output
        The output of the DepthMerger node containing spatial detections.
    shrinkingFactor : float
        The percentage of the bounding box to shrink from each side before
        sampling depth.

    Usage
    -----
    depth_merger = pipeline.create(DepthMerger).build(
        output2d=nn.out,
        outputDepth=stereo.depth
    )
    """

    def __init__(self, shrinkingFactor: float = 0) -> None:
        super().__init__()

        # TODO: We should make it consistant and use either output or out - IMO out is preferred to match DAI
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.SpatialImgDetections, True)
            ]
        )

        self.shrinking_factor = shrinkingFactor
        self._logger.debug(
            f"DepthMerger initialized with shrinking_factor={shrinkingFactor}"
        )

    def build(
        self,
        output2d: dai.Node.Output,
        outputDepth: dai.Node.Output,
        calibData: dai.CalibrationHandler,
        depthAlignmentSocket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
        shrinkingFactor: float = 0,
    ) -> "DepthMerger":
        """Connect detection and depth streams and initialize spatial conversion.

        Parameters
        ----------
        output2d
            Upstream output producing 2D detections.
        outputDepth
            Upstream output producing aligned depth frames.
        calibData
            Device calibration used to convert image coordinates into spatial
            coordinates.
        depthAlignmentSocket
            Camera socket the depth frame is aligned to.
        shrinkingFactor
            Percentage of each bounding box edge trimmed before depth averaging.

        Returns
        -------
        DepthMerger
            The configured node instance.
        """
        self.link_args(output2d, outputDepth)
        self.shrinking_factor = shrinkingFactor
        self.host_spatials_calc = HostSpatialsCalc(calibData, depthAlignmentSocket)
        self._logger.debug(
            f"DepthMerger built with shrinking_factor={shrinkingFactor}"
        )
        return self

    def process(self, message2d: dai.Buffer, depth: dai.ImgFrame) -> None:
        """Merge incoming detections with depth to produce spatial detections."""
        self._logger.debug("Processing new input")
        spatial_dets = self._transform(message2d, depth)
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
        else:
            raise ValueError(f"Unknown message type: {type(message_2d)}")

    def _detection_to_spatial(
        self,
        detection: dai.ImgDetection,
        depth: dai.ImgFrame,
    ) -> dai.SpatialImgDetection:
        """Converts a single 2D detection into a spatial detection using the depth
        frame."""
        depth_frame = depth.getCvFrame()
        x_len = depth_frame.shape[1]
        y_len = depth_frame.shape[0]
        xmin = detection.xmin
        ymin = detection.ymin
        xmax = detection.xmax
        ymax = detection.ymax
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
        spatials = self.host_spatials_calc.calcSpatials(depth, roi)

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
        spatial_img_detection.labelName = detection.labelName

        if isinstance(detection, dai.ImgDetection):
            kpts_transformed = []
            for kp in detection.getKeypoints():
                # spatialCoordinates are not computed per keypoint, only the
                # bounding box spatial coords are computed.
                skp = dai.Keypoint(
                    coordinates=kp.imageCoordinates,
                    confidence=kp.confidence,
                    label=kp.label,
                    labelName=kp.labelName,
                )
                kpts_transformed.append(skp)
            spatial_img_detection.setKeypoints(kpts_transformed)
            spatial_img_detection.setEdges(detection.getEdges())

        else:
            raise ValueError(
                f"Unknown detection type: {type(detection)}, expected dai.ImgDetection"
            )

        return spatial_img_detection

    def _detections_to_spatial(
        self,
        detections: dai.ImgDetections,
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
        bounded_coord = min(1., relative_coord)
        return max(0, int(bounded_coord * dimension_len) - 1)
