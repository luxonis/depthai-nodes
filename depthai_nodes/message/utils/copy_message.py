import copy

import depthai as dai


def copy_message(msg: dai.Buffer) -> dai.Buffer:
    """Copies the incoming message and returns it.

    @param msg: The input message.
    @type msg: dai.Buffer
    @return: The copied message.
    @rtype: dai.Buffer
    """

    # 1st attempt: native .copy() method
    if hasattr(msg, "copy"):
        return msg.copy()

    # 2nd attempt: custom copy implementation
    try:
        return _copy(msg)
    except TypeError:
        pass

    # 3rd attempt: deepcopy (the most general approach)
    try:
        return copy.deepcopy(msg)
    except TypeError as e:
        raise TypeError(f"Copying of message type {type(msg)} is not supported.") from e


def _copy(msg: dai.Buffer) -> dai.Buffer:
    def _copy_metadata(msg: dai.Buffer) -> dai.Buffer:
        msg_type = type(msg)
        msg_copy = msg_type()
        if hasattr(msg, "getSequenceNum"):
            msg_copy.setSequenceNum(msg.getSequenceNum())
        if hasattr(msg, "getTimestamp"):
            msg_copy.setTimestamp(msg.getTimestamp())
        if hasattr(msg, "getTimestampDevice"):
            msg_copy.setTimestampDevice(msg.getTimestampDevice())
        if hasattr(msg, "getTransformation"):
            transformation = msg.getTransformation()
            if transformation is not None:
                msg_copy.setTransformation(transformation)
        return msg_copy

    def _copy_img_frame(img_frame: dai.ImgFrame) -> dai.ImgFrame:
        img_frame_copy = _copy_metadata(img_frame)
        img_frame_copy.setCvFrame(img_frame.getCvFrame(), img_frame.getType())
        img_frame_copy.setCategory(img_frame.getCategory())
        return img_frame_copy

    def _copy_segmentation_mask(
        segmentation_mask: dai.SegmentationMask,
    ) -> dai.SegmentationMask:
        segmentation_mask_copy = _copy_metadata(segmentation_mask)
        assert isinstance(segmentation_mask_copy, dai.SegmentationMask)
        segmentation_mask_copy.setCvMask(segmentation_mask.getCvMask().copy())
        segmentation_mask_copy.setLabels(segmentation_mask.getLabels())
        return segmentation_mask_copy

    def _copy_img_detection(
        img_det: dai.ImgDetection | dai.SpatialImgDetection,
    ) -> dai.ImgDetection | dai.SpatialImgDetection:
        assert isinstance(img_det, (dai.ImgDetection, dai.SpatialImgDetection))
        img_det_copy = _copy_metadata(img_det)
        assert isinstance(img_det_copy, (dai.ImgDetection, dai.SpatialImgDetection))
        if isinstance(img_det, dai.SpatialImgDetection):
            img_det_copy.spatialCoordinates = img_det.spatialCoordinates
            img_det_copy.boundingBoxMapping = img_det.boundingBoxMapping
        img_det_copy.label = img_det.label
        img_det_copy.labelName = img_det.labelName
        img_det_copy.confidence = img_det.confidence
        img_det_copy.setBoundingBox(_copy_rotated_rect(img_det.getBoundingBox()))
        img_det_copy.setKeypoints(_copy_keypoints(img_det.getKeypoints()))
        img_det_copy.setEdges(copy.deepcopy(img_det.getEdges()))
        return img_det_copy

    def _copy_img_detections(
        img_dets: dai.ImgDetections | dai.SpatialImgDetections,
    ) -> dai.ImgDetections | dai.SpatialImgDetections:
        assert isinstance(img_dets, (dai.ImgDetections, dai.SpatialImgDetections))
        img_dets_copy = _copy_metadata(img_dets)
        if isinstance(img_dets, dai.ImgDetections):
            assert isinstance(img_dets_copy, dai.ImgDetections)
            masks = img_dets.getCvSegmentationMask()
            if masks is not None:
                img_dets_copy.setCvSegmentationMask(masks)
        img_dets_copy.detections = [
            _copy_img_detection(img_det) for img_det in img_dets.detections
        ]
        return img_dets_copy

    def _copy_keypoints(keypoints: list[dai.Keypoint]) -> list[dai.Keypoint]:
        keypoints_copy = [_copy_keypoint(keypoint) for keypoint in keypoints]
        return keypoints_copy

    def _copy_keypoints_list(keypoints: dai.KeypointsList) -> dai.KeypointsList:
        keypoints_copy = _copy_metadata(keypoints)
        assert isinstance(keypoints_copy, dai.KeypointsList)
        keypoints_copy.setKeypoints(_copy_keypoints(keypoints.getKeypoints()))
        keypoints_copy.setEdges(copy.deepcopy(keypoints.getEdges()))
        return keypoints_copy

    def _copy_point2f(point2f: dai.Point2f) -> dai.Point2f:
        point2f_copy = _copy_metadata(point2f)
        point2f_copy.x = point2f.x
        point2f_copy.y = point2f.y
        return point2f_copy

    def _copy_size2f(size2f: dai.Size2f) -> dai.Size2f:
        size2f_copy = _copy_metadata(size2f)
        size2f_copy.width = size2f.width
        size2f_copy.height = size2f.height
        return size2f_copy

    def _copy_point3f(point3f: dai.Point3f) -> dai.Point3f:
        point3f_copy = _copy_metadata(point3f)
        point3f_copy.x = point3f.x
        point3f_copy.y = point3f.y
        point3f_copy.z = point3f.z
        return point3f_copy

    def _copy_keypoint(keypoint: dai.Keypoint) -> dai.Keypoint:
        keypoint_copy = _copy_metadata(keypoint)
        keypoint_copy.imageCoordinates = _copy_point3f(keypoint.imageCoordinates)
        keypoint_copy.confidence = keypoint.confidence
        keypoint_copy.label = keypoint.label
        keypoint_copy.labelName = keypoint.labelName
        return keypoint_copy

    def _copy_rotated_rect(rotated_rect: dai.RotatedRect) -> dai.RotatedRect:
        rotated_rect_copy: dai.RotatedRect = _copy_metadata(rotated_rect)
        rotated_rect_copy.center = _copy_point2f(rotated_rect.center)
        rotated_rect_copy.size = _copy_size2f(rotated_rect.size)
        rotated_rect_copy.angle = rotated_rect.angle
        return rotated_rect_copy

    def _copy_beta_message(msg: dai.Buffer) -> dai.Buffer:
        msg_copy = _copy_metadata(msg)

        if isinstance(msg, dai.beta.Classifications):
            msg_copy.classes = list(msg.classes)
            msg_copy.scores = msg.scores.copy()
        elif isinstance(msg, dai.beta.Clusters):
            clusters = []
            for cluster in msg.clusters:
                cluster_copy = dai.beta.Cluster()
                cluster_copy.label = cluster.label
                cluster_copy.points = dai.VectorPoint2f(
                    [_copy_point2f(point) for point in cluster.points]
                )
                clusters.append(cluster_copy)
            msg_copy.clusters = clusters
        elif isinstance(msg, dai.beta.Keypoints):
            msg_copy.setKeypoints(
                _copy_keypoints(msg.getKeypoints()), copy.deepcopy(msg.getEdges())
            )
        elif isinstance(msg, dai.beta.Lines):
            lines = []
            for line in msg.lines:
                line_copy = dai.beta.Line()
                line_copy.startPoint = _copy_point2f(line.startPoint)
                line_copy.endPoint = _copy_point2f(line.endPoint)
                line_copy.confidence = line.confidence
                lines.append(line_copy)
            msg_copy.lines = lines
        elif isinstance(msg, dai.beta.Map2D):
            map_array = msg.getMap()
            if map_array.size > 0:
                msg_copy.setMap(map_array.copy())
        elif isinstance(msg, dai.beta.Predictions):
            predictions = []
            for prediction in msg.predictions:
                prediction_copy = dai.beta.Prediction()
                prediction_copy.prediction = prediction.prediction
                predictions.append(prediction_copy)
            msg_copy.predictions = predictions
        else:
            raise TypeError(f"Unsupported beta message type {type(msg)}")

        return msg_copy

    if isinstance(msg, dai.SegmentationMask):
        return _copy_segmentation_mask(msg)
    elif isinstance(msg, dai.ImgFrame):
        return _copy_img_frame(msg)
    elif isinstance(msg, (dai.ImgDetection, dai.SpatialImgDetection)):
        return _copy_img_detection(msg)
    elif isinstance(msg, (dai.ImgDetections, dai.SpatialImgDetections)):
        return _copy_img_detections(msg)
    elif isinstance(msg, dai.KeypointsList):
        return _copy_keypoints_list(msg)
    elif isinstance(msg, dai.Point2f):
        return _copy_point2f(msg)
    elif isinstance(
        msg,
        (
            dai.beta.Classifications,
            dai.beta.Clusters,
            dai.beta.Keypoints,
            dai.beta.Lines,
            dai.beta.Map2D,
            dai.beta.Predictions,
        ),
    ):
        return _copy_beta_message(msg)
    else:
        # TODO: define logic for copying other message types
        raise TypeError(f"Copying of message type {type(msg)} is not supported.")
