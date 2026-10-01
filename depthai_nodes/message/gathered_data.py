from typing import Generic, TypeVar

import depthai as dai

from .collection import Collection

TReference = TypeVar("TReference", bound=dai.Buffer)
TGathered = TypeVar("TGathered", bound=dai.Buffer)


class GatheredData(Collection[TGathered], Generic[TReference, TGathered]):
    """Contains N messages and reference data that the messages were matched with.

    Attributes:
        reference_data (``TReference``): Data that is used to determine how many of
            TGathered to gather.
        items (``list[TGathered]``): List of gathered data.
    """

    def __init__(self, reference_data: TReference, items: list[TGathered]) -> None:
        """Initialize gathered items and copy metadata from the reference message.

        Args:
            reference_data: Reference whose timestamps and sequence number are copied.
            items: Messages gathered for this reference, all of the same runtime type.
        """
        super().__init__(items=items)
        self.reference_data = reference_data

    @property
    def reference_data(self) -> TReference:
        """Returns the reference data.

        Returns:
            Reference data.
        """
        return self._reference_data

    @reference_data.setter
    def reference_data(self, value: TReference):
        """Sets the reference data.

        Args:
            value: Reference data.
        """
        self.setSequenceNum(value.getSequenceNum())
        self.setTimestamp(value.getTimestamp())
        self.setTimestampDevice(value.getTimestampDevice())
        self._reference_data = value
