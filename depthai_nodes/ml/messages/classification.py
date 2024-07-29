from typing import List

import depthai as dai


class Classifications(dai.Buffer):
    def __init__(self):
        dai.Buffer.__init__(self)
        self._classes = []

    @property
    def classes(self) -> List:
        return self._classes

    @classes.setter
    def classes(self, value: List):
        if not isinstance(value, list):
            raise TypeError("Must be a list.")
        for item in value:
            if not isinstance(item, list) or len(item) != 2:
                raise TypeError(
                    "Each item must be a list of [class_name, probability_score], got {item}."
                )
        self._classes = value
