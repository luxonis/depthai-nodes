from typing import Generic, List, TypeVar

import depthai as dai

TReference = TypeVar("TReference", bound=dai.Buffer)
TGathered = TypeVar("TGathered")


class GatheredData(dai.Buffer, Generic[TReference, TGathered]):
    """Contains N messages and reference data that the messages were matched with.

    Attributes
    ----------
    reference_data: TReference
        Data that is used to determine how many of TGathered to gather.
    collected: List[TGathered]
        List of collected data.
    """

    def __init__(self, reference_data: TReference, gathered: List[TGathered]) -> None:
        """Initializes the GatheredData object."""
        super().__init__()
        self.reference_data = reference_data
        self.gathered = gathered

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
        self.setSequenceNum(value.getSequenceNum())
        self.setTimestamp(value.getTimestamp())
        self.setTimestampDevice(value.getTimestampDevice())
        self._reference_data = value

    @property
    def gathered(self) -> List[TGathered]:
        """Returns the collected data.

        @return: List of collected data.
        @rtype: List[TGathered]
        """
        return self._gathered

    @gathered.setter
    def gathered(self, value: List[TGathered]):
        """Sets the gathered data.

        @param value: List of gathered data.
        @type value: List[TGathered]
        @raise TypeError: If value is not a list.
        """
        if not isinstance(value, list):
            raise TypeError("gathered_data must be a list.")
        self._gathered = value
