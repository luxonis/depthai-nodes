from typing import Dict, List, Union

import depthai as dai


class CompositeMessage(dai.Buffer):
    """CompositeMessage class for storing composite of (dai.Buffer, float, List) data.

    Attributes
    ----------
    _data : Dict[str, Union[dai.Buffer, float, List]]
        Dictionary of data with keys as string and values as either dai.Buffer, float or List.
    """

    def __init__(self):
        super().__init__()
        self._data: Dict[str, Union[dai.Buffer, float, List]] = {}

    def setData(self, data: Dict[str, Union[dai.Buffer, float, List]]):
        self._data = data

    def getData(self) -> Dict[str, Union[dai.Buffer, float, List]]:
        return self._data
