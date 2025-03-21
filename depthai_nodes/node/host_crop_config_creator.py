from typing import Optional

import depthai as dai
import numpy as np

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended


class CropConfigsCreatorNode(dai.node.HostNode):
    """This node is used to create a dai.ImageManipConfigV2() object for every detection
    in a ImgDetectionsExtended message. The node iterates over a list of n detections
    and sends a dai.ImgeManipConfigV2 objects for each detection. By default, the node
    will keep at most the first 100 detections.

    Before use, the source and target image sizes need to be set with the build function.
    The node assumes the last frame is saved in the dai.ImgManipV2 node and when recieving a detection_message the node sends an empty crop config that skips the frame and loads the next frame in the queue.

    Attributes
    ----------
    detections_input : dai.Input
        The input link for the ImgDetectionsExtended message
    config_output : dai.Output
        The output link for the ImageManipConfigV2 messages
    detections_output : dai.Output
        The output link for the ImgDetectionsExtended message
    w : int
        The width of the source image.
    h : int
        The height of the source image.
    target_w : int
        The width of the target image.
    target_h : int
        The height of the target image.
    n_detections : int
        The number of detections to keep.
    """

    def __init__(self) -> None:
        """Initializes the node."""
        super().__init__()
        self.config_output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfigV2, True)
            ]
        )
        self.detections_output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )
        print(self.detections_output)
        print(self.config_output)
        self._w: Optional[int] = None
        self._h: Optional[int] = None
        self._target_w: Optional[int] = None
        self._target_h: Optional[int] = None
        self._n_detections: Optional[int] = None

    @property
    def w(self) -> Optional[int]:
        """Returns the width of the source image.

        @return: Width of the source image.
        @rtype: int
        """
        return self._w

    @property
    def h(self) -> Optional[int]:
        """Returns the height of the source image.

        @return: Height of the source image.
        @rtype: int
        """
        return self._h

    @property
    def target_w(self) -> Optional[int]:
        """Returns the width of the target image.

        @return: Width of the target image.
        @rtype: int
        """
        return self._target_w

    @property
    def target_h(self) -> Optional[int]:
        """Returns the height of the target image.

        @return: Height of the target image.
        @rtype: int
        """
        return self._target_h

    @property
    def n_detections(self) -> Optional[int]:
        """Returns the number of detections to keep.

        @return: Number of detections to keep.
        @rtype: int
        """
        return self._n_detections

    @w.setter
    def w(self, w: int):
        """Sets the width of the source image.

        @param w: Width of the source image.
        @type w: int
        @raise TypeError: If w is not an integer.
        @raise ValueError: If w is less than 1.
        """
        self._validate_positive_integer(w)
        self._w = w

    @h.setter
    def h(self, h: int):
        """Sets the height of the source image.

        @param h: Height of the source image.
        @type h: int
        @raise TypeError: If h is not an integer.
        @raise ValueError: If h is less than 1.
        """
        self._validate_positive_integer(h)
        self._h = h

    @target_w.setter
    def target_w(self, target_w: int):
        """Sets the width of the target image.

        @param target_w: Width of the target image.
        @type target_w: int
        @raise TypeError: If target_w is not an integer.
        @raise ValueError: If target_w is less than 1.
        """
        self._validate_positive_integer(target_w)
        self._target_w = target_w

    @target_h.setter
    def target_h(self, target_h: int):
        """Sets the height of the target image.

        @param target_h: Height of the target image.
        @type target_h: int
        @raise TypeError: If target_h is not an integer.
        @raise ValueError: If target_h is less than 1.
        """
        self._validate_positive_integer(target_h)
        self._target_h = target_h

    @n_detections.setter
    def n_detections(self, n_detections: int):
        """Sets the number of detections to keep.

        @param n_detections: Number of detections to keep.
        @type n_detections: int
        @raise TypeError: If n_detections is not an integer.
        @raise ValueError: If n_detections is less than 1.
        """
        self._validate_positive_integer(n_detections)
        self._n_detections = n_detections

    def build(
        self,
        detections_input: dai.Node.Output,
        w: int,
        h: int,
        target_w: int,
        target_h: int,
        n_detections: int = 100,
    ) -> "CropConfigsCreatorNode":
        """Link the node input and set the correct source and target image sizes.

        Parameters
        ----------
        detections_input : dai.Node.Output
            The input link for the ImgDetectionsExtended message
        w : int
            The width of the source image.
        h : int
            The height of the source image.
        target_w : int
            The width of the target image.
        target_h : int
            The height of the target image.
        n_detections : int, optional
            The number of detections to keep, by default 100
        """
        self.w = w
        self.h = h
        self.target_w = target_w
        self.target_h = target_h
        self.n_detections = n_detections
        self.link_args(detections_input)

        return self

    def process(self, detections_input: dai.Buffer) -> None:
        """Process the input detections and create crop configurations. This function is
        ran every time a new ImgDetectionsExtended message is received.

        Sends the first n detections to the detections_output link and crop
        configurations to the config_output link. The first crop config has the
        setReusePreviousImage flag set to False, which changes the previous frame for a
        new one.
        """
        assert isinstance(detections_input, ImgDetectionsExtended)
        detections = detections_input.detections
        sequence_num = detections_input.getSequenceNum()
        timestamp = detections_input.getTimestamp()

        detections_to_keep = []
        num_detections = min(len(detections), self._n_detections)

        # Skip the current frame / load new frame
        cfg = dai.ImageManipConfigV2()
        cfg.setSkipCurrentImage(True)
        cfg.setTimestamp(timestamp)
        cfg.setSequenceNum(sequence_num)
        send_status = False
        while not send_status:
            send_status = self.config_output.trySend(cfg)

        for i in range(num_detections):
            cfg = dai.ImageManipConfigV2()
            detection: ImgDetectionExtended = detections[i]
            detections_to_keep.append(detection)
            rect = detection.rotated_rect
            rect = rect.denormalize(self.w, self.h)

            cfg.addCropRotatedRect(rect, normalizedCoords=False)
            cfg.setOutputSize(self.target_w, self.target_h)
            cfg.setReusePreviousImage(True)
            cfg.setTimestamp(timestamp)
            cfg.setSequenceNum(sequence_num)

            send_status = False
            while not send_status:
                send_status = self.config_output.trySend(cfg)

        detections_msg = ImgDetectionsExtended()
        detections_msg.setSequenceNum(sequence_num)
        detections_msg.setTimestamp(timestamp)
        detections_msg.setTransformation(detections_input.getTransformation())
        detections_msg.detections = detections_to_keep

        if detections_input.masks.ndim == 2:
            masks = np.where(
                detections_input.masks >= num_detections, -1, detections_input.masks
            )
            detections_msg.masks = masks
        self.detections_output.send(detections_input)

    def _validate_positive_integer(self, value: int):
        """Validates that the set size is a positive integer.

        @param value: The value to validate.
        @type value: int
        @raise TypeError: If value is not an integer.
        @raise ValueError: If value is less than 1.
        """
        if not isinstance(value, int):
            raise TypeError("Value must be an integer.")
        if value < 1:
            raise ValueError("Value must be greater than 1.")
