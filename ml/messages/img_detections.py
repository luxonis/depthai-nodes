import depthai as dai
from typing import List, Tuple, Union


class ImgDetectionWithKeypoints(dai.ImgDetection):
    def __init__(self):
        dai.ImgDetection.__init__(self)  # TODO: change to super().__init__()?
        self.keypoints: List[Tuple[float, float]] = []

    @property
    def keypoints(self) -> List[Tuple[float, float]]:
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value: List[Tuple[Union[int, float], Union[int, float]]]):
        if not isinstance(value, list):
            raise TypeError("Keypoints must be a list")
        for item in value:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(i, (int, float)) for i in item)
            ):
                raise TypeError(
                    "Each keypoint must be a tuple of two floats or integers"
                )
        self._keypoints = [(float(x), float(y)) for x, y in value]


class DetectionsWithKeypointsDescriptor:
    """
    Descriptor for managing the `detections` attribute with type checking.

    This descriptor ensures that the `detections` attribute is a list of
    ImgDetectionWithKeypoints instances.

    Attributes:
        name (str): The name of the attribute managed by the descriptor.

    Methods:
        __get__(instance, owner):
            Retrieves the value of the `detections` attribute.
        __set__(instance, value):
            Sets the value of the `detections` attribute after validating its type.
    """

    def __init__(self):
        self.name = "_detections"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, [])

    def __set__(self, instance, value: List["ImgDetectionWithKeypoints"]):
        if not isinstance(value, list):
            raise TypeError("Detections must be a list")
        for item in value:
            if not isinstance(item, ImgDetectionWithKeypoints):
                raise TypeError(
                    "Each detection must be an instance of ImgDetectionWithKeypoints"
                )
        instance.__dict__[self.name] = value


class ImgDetectionsWithKeypoints(dai.Buffer):
    detections = DetectionsWithKeypointsDescriptor()

    def __init__(self):
        dai.Buffer.__init__(self) # TODO: change to super().__init__()?
        self.detections: List[ImgDetectionWithKeypoints] = []
