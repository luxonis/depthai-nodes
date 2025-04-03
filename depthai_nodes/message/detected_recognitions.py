from typing import List, Union, Optional

import depthai as dai

from .img_detections import ImgDetectionsExtended


class DetectedRecognitions(dai.Buffer):
    """A class for storing image detections combined with neural network data.

    Attributes
    ----------
    img_detections: Union[dai.ImgDetections, ImgDetectionsExtended]
        Image detections with keypoints and masks.
    nn_data: List[dai.NNData]
        Neural network output data corresponding to the detections.
    """

    def __init__(self) -> None:
        """Initializes the DetectedRecognitions object."""
        super().__init__()
        self._img_detections = None
        self._nn_data = []

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
    def nn_data(self) -> Optional[List[dai.NNData]]:
        """Returns the neural network data.

        @return: List of neural network data.
        @rtype: List[dai.NNData]
        """
        return self._nn_data

    @nn_data.setter
    def nn_data(self, value: Optional[List[dai.NNData]]):
        """Sets the neural network data.

        @param value: List of neural network data.
        @type value: Optional[List[dai.NNData]]
        @raise TypeError: If value is not a list.
        @raise TypeError: If each element is not a dai.NNData object.
        """
        if value is None:
            self._nn_data = []
            return

        if not isinstance(value, list):
            raise TypeError("nn_data must be a list.")
        self._nn_data = value
