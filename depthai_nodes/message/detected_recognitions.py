from typing import List, Optional, Union

import depthai as dai

from .img_detections import ImgDetectionsExtended


class DetectedRecognitions(dai.Buffer):
    """A class for storing image detections combined with recognitions data.

    Attributes
    ----------
    img_detections: Union[dai.ImgDetections, ImgDetectionsExtended]
        Image detections with keypoints and masks.
    recognitions_data: List[dai.Buffer]
        List of neural network data.
    """

    def __init__(self) -> None:
        """Initializes the DetectedRecognitions object."""
        super().__init__()
        self._img_detections = None
        self._recognitions_data = []

    @property
    def img_detections(self) -> Union[dai.ImgDetections, ImgDetectionsExtended]:
        """Returns the image detections.

        @return: Image detections with keypoints and masks.
        @rtype: Union[dai.ImgDetections, ImgDetectionsExtended]
        """
        return self._img_detections

    @img_detections.setter
    def img_detections(self, value: Union[dai.ImgDetections, ImgDetectionsExtended]):
        """Sets the image detections.

        @param value: Image detections with keypoints and masks.
        @type value: Union[dai.ImgDetections, ImgDetectionsExtended]
        @raise TypeError: If value is not an ImgDetections or ImgDetectionsExtended
            object.
        """
        if not isinstance(value, (dai.ImgDetections, ImgDetectionsExtended)):
            raise TypeError(
                "img_detections must be an ImgDetections or ImgDetectionsExtended object."
            )
        self._img_detections = value
        self.setTimestampDevice(value.getTimestampDevice())
        self.setTimestamp(value.getTimestamp())
        self.setSequenceNum(value.getSequenceNum())

    @property
    def recognitions_data(self) -> Optional[List[dai.Buffer]]:
        """Returns the recognitions data.

        @return: List of recognitions data.
        @rtype: Optional[List[dai.Buffer]]
        """
        return self._recognitions_data

    @recognitions_data.setter
    def recognitions_data(self, value: Optional[List[dai.Buffer]]):
        """Sets the recognitions data.

        @param value: List of recognitions data.
        @type value: Optional[List[dai.Buffer]]
        @raise TypeError: If value is not a list.
        @raise TypeError: If each element is not of type dai.Buffer.
        """
        if value is None:
            self._recognitions_data = []
            return

        if not isinstance(value, list):
            raise TypeError("recognitions_data must be a list.")
        if not all(isinstance(item, dai.Buffer) for item in value):
            raise TypeError("recognitions_data must be a list of dai.Buffer objects.")
        self._recognitions_data = value
