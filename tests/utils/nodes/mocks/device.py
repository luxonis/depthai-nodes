import depthai as dai
import numpy as np


class DeviceMock:
    """A device with a settable ID, platform and active calibration.

    Its calibration starts empty, without camera data or IMU extrinsics.
    """

    def __init__(self, deviceId: str = "default"):
        self._device_id = deviceId
        self._platform = dai.Platform.RVC2
        self._calibration = dai.CalibrationHandler()

    def getDeviceId(self):
        return self._device_id

    def getPlatformAsString(self):
        return self._platform.name

    def getPlatform(self):
        return self._platform

    def setPlatform(self, platform: dai.Platform):
        self._platform = platform

    def getCalibration(self) -> dai.CalibrationHandler:
        return self._calibration

    def setCalibration(self, calibration: dai.CalibrationHandler):
        self._calibration = calibration


def camera_calibration(
    socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
) -> dai.CalibrationHandler:
    """A calibration with intrinsics for ``socket`` and no IMU extrinsics."""
    calibration = dai.CalibrationHandler()
    calibration.setCameraIntrinsics(socket, np.eye(3).tolist(), 640, 400)
    return calibration


def imu_calibration(
    rotation: np.ndarray,
    socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
) -> dai.CalibrationHandler:
    """A calibration whose IMU → ``socket`` extrinsics rotate by ``rotation``.

    DepthAI reads IMU extrinsics only for a camera it has data for, so the camera's
    intrinsics are set first.
    """
    calibration = camera_calibration(socket)
    calibration.setImuExtrinsics(socket, np.asarray(rotation).tolist(), [0, 0, 0])
    return calibration


def imu_device(
    rotation: np.ndarray | None = None, deviceId: str = "1000"
) -> DeviceMock:
    """A device whose IMU → CAM_A extrinsics rotate by ``rotation``.

    ``None`` means the identity.
    """
    device = DeviceMock(deviceId)
    device.setCalibration(imu_calibration(np.eye(3) if rotation is None else rotation))
    return device
