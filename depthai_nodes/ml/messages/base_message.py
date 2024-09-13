from typing import Dict, List, Union

import depthai as dai


class MiscelaneousMessage(dai.Buffer):
    def __init__(self):
        super().__init__()
        self._data: Dict[str, Union[dai.Buffer, float, List]] = {}

    def setData(self, data: Dict[str, Union[dai.Buffer, float, List]]):
        self._data = data

    def getData(self) -> Dict[str, Union[dai.Buffer, float, List]]:
        return self._data
