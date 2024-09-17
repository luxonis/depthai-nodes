from typing import List

import depthai as dai


class Keypoints(dai.Buffer):
    """Keypoints class for storing keypoints.

    Attributes
    ----------
    keypoints: List[dai.Point3f]
        List of dai.Point3f, each representing a keypoint.
    """

    def __init__(self):
        """Initializes the Keypoints object."""
        super().__init__()
        self._keypoints: List[dai.Point3f] = []

    @property
    def keypoints(self) -> List[dai.Point3f]:
        """Returns the keypoints.

        @return: List of keypoints.
        @rtype: List[dai.Point3f]
        """
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value: List[dai.Point3f]):
        """Sets the keypoints.

        @param value: List of keypoints.
        @type value: List[dai.Point3f]
        @raise TypeError: If the keypoints are not a list.
        @raise TypeError: If each keypoint is not of type dai.Point3f.
        """
        if not isinstance(value, list):
            raise TypeError("keypoints must be a list.")
        for item in value:
            if not isinstance(item, dai.Point3f):
                raise TypeError("All items in keypoints must be of type dai.Point3f.")
        self._keypoints = value


class KeypointsWithObjectness(Keypoints):
    """KeypointsWithObjectness class for storing keypoints with objectness score.

    Attributes
    ----------
    keypoints: List[dai.Point3f]
        List of dai.Point3f, each representing a keypoint.
    objectness: float
        Objectness score indicating the presence of an object in image.
    """

    def __init__(self):
        """Initializes the KeypointsWithObjectness object."""
        Keypoints.__init__(self)
        self._objectness: float = 0.0

    @property
    def objectness(self) -> float:
        """Returns the objectness.

        @return: objectness score indicating the presence of an object in image.
        @rtype: float
        """
        return self._objectness

    @objectness.setter
    def objectness(self, value: float):
        """Sets the objectness score.

        @param value: objectness score indicating the presence of an object in image.
        @type value: float
        @raise TypeError: If the objectness is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("objectness must be a float.")
        self._objectness = value


class HandKeypoints(KeypointsWithObjectness):
    """HandKeypoints class for storing hand keypoints.

    Attributes
    ----------
    keypoints: List[dai.Point3f]
        List of dai.Point3f, each representing a keypoint.
    objectness: float
        Objectness score indicating the presence of an object in image.
    handedness: float
        Handedness of the hand. 0.0 for left hand and 1.0 for right hand.
    """

    def __init__(self):
        """Initializes the HandKeypoints object."""
        KeypointsWithObjectness.__init__(self)
        self._handedness: float = 0.0

    @property
    def handedness(self) -> float:
        """Returns the handedness of the hand.

        @return: Handedness of the hand.
        @rtype: float
        """
        return self._handedness

    @handedness.setter
    def handedness(self, value: float):
        """Sets the handedness of the hand.

        @param value: Handedness of the hand.
        @type value: float
        @raise TypeError: If the handedness is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("handedness must be a float.")
        self._handedness = value
