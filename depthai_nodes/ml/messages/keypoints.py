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


class HandKeypoints(Keypoints):
    """HandKeypoints class for storing hand keypoints.

    Attributes
    ----------
    confidence: float
        Confidence of the hand keypoints.
    handdedness: float
        Handedness of the hand keypoints. 0.0 for left hand and 1.0 for right hand.
    """

    def __init__(self):
        """Initializes the HandKeypoints object."""
        Keypoints.__init__(self)
        self._confidence: float = 0.0
        self._handdedness: float = 0.0

    @property
    def confidence(self) -> float:
        """Returns the confidence of the hand keypoints.

        @return: Confidence of the hand keypoints.
        @rtype: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value: float):
        """Sets the confidence of the hand keypoints.

        @param value: Confidence of the hand keypoints.
        @type value: float
        @raise TypeError: If the confidence is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("confidence must be a float.")
        self._confidence = value

    @property
    def handdedness(self) -> float:
        """Returns the handdedness of the hand keypoints.

        @return: Handdedness of the hand keypoints.
        @rtype: float
        """
        return self._handdedness

    @handdedness.setter
    def handdedness(self, value: float):
        """Sets the handdedness of the hand keypoints.

        @param value: Handdedness of the hand keypoints.
        @type value: float
        @raise TypeError: If the handdedness is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("handdedness must be a float.")
        self._handdedness = value
