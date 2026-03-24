from typing import Generic, List, TypeVar

import depthai as dai

from .collection import Collection

TReference = TypeVar("TReference", bound=dai.Buffer)
TGathered = TypeVar("TGathered", bound=dai.Buffer)


class GatheredData(Collection[TGathered], Generic[TReference, TGathered]):
    """Contains N messages and reference data that the messages were matched with.

    Attributes
    ----------
    reference_data: TReference
        Data that is used to determine how many of TGathered to gather.
    items: List[TGathered]
        List of gathered data.
    """

    def __init__(self, reference_data: TReference, items: List[TGathered]) -> None:
        """Initializes the GatheredData object."""
        super().__init__(items=items)
        self.reference_data = reference_data

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
