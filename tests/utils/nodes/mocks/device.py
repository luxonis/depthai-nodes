import depthai as dai


class DeviceMock:
    def __init__(self, deviceId: str = "default"):
        self._device_id = deviceId
        self._platform = dai.Platform.RVC2

    def getDeviceId(self):
        return self._device_id

    def getPlatformAsString(self):
        return self._platform.name

    def getPlatform(self):
        return self._platform

    def setPlatform(self, platform: dai.Platform):
        self._platform = platform
