import depthai as dai


class DeviceMock:
    def __init__(self):
        self._platform = dai.Platform(0)  # initialize platform with 0 (RVC2)

    def getPlatformAsString(self):
        return self._platform.name

    def getPlatform(self):
        return self._platform
