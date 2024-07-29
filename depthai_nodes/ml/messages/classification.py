from typing import List

import depthai as dai


class ClassificationMessage(dai.Buffer):
    def __init__(self):
        dai.Buffer.__init__(self)
        self._sortedClasses = []

    @property
    def sortedClasses(self) -> List:
        return self._sortedClasses

    @sortedClasses.setter
    def sortedClasses(self, value: List):
        if not isinstance(value, list):
            raise TypeError("Sorted classes must be a list.")
        for item in value:
            if not isinstance(item, list) or len(item) != 2:
                raise TypeError("Each sorted class must be a list of 2 elements.")
            if not isinstance(item[0], str):
                raise TypeError("Class name must be a string.")
        self._sortedClasses = value
