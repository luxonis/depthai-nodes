from typing import Generic, List, TypeVar

import depthai as dai

TReference = TypeVar("TReference")
TCollected = TypeVar("TCollected")


class GatheredData(dai.Buffer, Generic[TReference, TCollected]):
    """A class for gathered number of data and the reference data on which the data was
    gathered.

    Attributes
    ----------
    reference_data: TReference
        Data that is used to determine how many of TCollected to gather.
    collected: List[TCollected]
        List of collected data.
    """

    def __init__(self, reference_data: TReference, collected: List[TCollected]) -> None:
        """Initializes the DetectedRecognitions object."""
        super().__init__()
        self._reference_data = reference_data
        self._collected = collected

    @property
    def reference_data(self) -> TReference:
        """Returns the reference data.

        @return: Reference data.
        @rtype: TReference
        """
        return self._reference_data

    @reference_data.setter
    def reference_data(self, value: TReference):
        """Sets the reference data.

        @param value: Reference data.
        @type value: TReference
        """
        self._reference_data = value

    @property
    def collected(self) -> List[TCollected]:
        """Returns the collected data.

        @return: List of collected data.
        @rtype: List[TCollected]
        """
        return self._collected

    @collected.setter
    def collected(self, value: List[TCollected]):
        """Sets the collected data.

        @param value: List of collected data.
        @type value: List[TCollected]
        @raise TypeError: If value is not a list.
        """
        if not isinstance(value, list):
            raise TypeError("collected_data must be a list.")
        self._collected = value
