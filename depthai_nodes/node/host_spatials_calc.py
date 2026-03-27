from typing import Callable, Dict, List

import depthai as dai
import numpy as np


class HostSpatialsCalc:
    """HostSpatialsCalc is a helper class for calculating spatial coordinates from depth
    data.

    Attributes
    ----------
    calibData : dai.CalibrationHandler
        Calibration data handler for the device.
    depthAlignmentSocket : dai.CameraBoardSocket
        The camera socket used for depth alignment.
    delta : int
        The delta value for ROI calculation. Default is 5 - means 10x10 depth pixels around point for depth averaging.
    threshLow : int
        The lower threshold for depth values. Default is 200 - means 20cm.
    threshHigh : int
        The upper threshold for depth values. Default is 30000 - means 30m.
    """

    # We need device object to get calibration data
    def __init__(
        self,
        calibData: dai.CalibrationHandler,
        depthAlignmentSocket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
        delta: int = 5,
        threshLow: int = 200,
        threshHigh: int = 30000,
    ):
        self.calibData = calibData
        self.depth_alignment_socket = depthAlignmentSocket

        self.delta = delta
        self.thresh_low = threshLow
        self.thresh_high = threshHigh

    def setLowerThreshold(self, thresholdLow: int) -> None:
        """Set the lower depth threshold used during ROI averaging.

        Parameters
        ----------
        thresholdLow
            Lower accepted depth value.
        """
        if not isinstance(thresholdLow, int):
            if isinstance(thresholdLow, float):
                thresholdLow = int(thresholdLow)
            else:
                raise TypeError(
                    "Threshold has to be an integer or float! Got {}".format(
                        type(thresholdLow)
                    )
                )
        self.thresh_low = thresholdLow

    def setUpperThreshold(self, thresholdHigh: int) -> None:
        """Set the upper depth threshold used during ROI averaging.

        Parameters
        ----------
        thresholdHigh
            Upper accepted depth value.
        """
        if not isinstance(thresholdHigh, int):
            if isinstance(thresholdHigh, float):
                thresholdHigh = int(thresholdHigh)
            else:
                raise TypeError(
                    "Threshold has to be an integer or float! Got {}".format(
                        type(thresholdHigh)
                    )
                )
        self.thresh_high = thresholdHigh

    def setDeltaRoi(self, delta: int) -> None:
        """Set the half-size of the ROI used around point inputs."""
        if not isinstance(delta, int):
            if isinstance(delta, float):
                delta = int(delta)
            else:
                raise TypeError(
                    "Delta has to be an integer or float! Got {}".format(type(delta))
                )
        self.delta = delta

    def calcSpatials(
        self,
        depthData: dai.ImgFrame,
        roi: List[int],
        averagingMethod: Callable = np.mean,
    ) -> Dict[str, float]:
        """Calculate spatial coordinates from the depth frame within the ROI.

        Parameters
        ----------
        depthData
            Depth frame used for coordinate estimation.
        roi
            Region of interest or point.
        averagingMethod
            Callable used to reduce valid depth values inside the ROI.

        Returns
        -------
        Dict[str, float]
            Spatial coordinates in camera space.
        """
        depthFrame = depthData.getFrame()

        roi = self._check_input(
            roi, depthFrame
        )  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.thresh_low <= depthROI) & (depthROI <= self.thresh_high)

        valid_depths = depthROI[inRange]
        if valid_depths.size == 0:
            return {
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
            }
        else:
            averageDepth = averagingMethod(valid_depths)

        centroid = np.array(  # Get centroid of the ROI
            [
                int((xmax + xmin) / 2),
                int((ymax + ymin) / 2),
            ]
        )

        K = self.calibData.getCameraIntrinsics(
            cameraId=self.depth_alignment_socket,
            resizeWidth=depthFrame.shape[1],
            resizeHeight=depthFrame.shape[0],
        )
        K = np.array(K)
        K_inv = np.linalg.inv(K)
        homogenous_coords = np.array([centroid[0], centroid[1], 1])
        spatial_coords = averageDepth * K_inv.dot(homogenous_coords)

        spatials = {
            "x": spatial_coords[0],
            "y": spatial_coords[1],
            "z": spatial_coords[2],
        }
        return spatials

    def _check_input(self, roi: List[int], frame: np.ndarray) -> List[int]:
        if len(roi) == 4:
            return roi
        if len(roi) != 2:
            raise ValueError(
                "You have to pass either ROI (4 values) or point (2 values)!"
            )
        x = min(max(roi[0], self.delta), frame.shape[1] - self.delta)
        y = min(max(roi[1], self.delta), frame.shape[0] - self.delta)
        return [x - self.delta, y - self.delta, x + self.delta, y + self.delta]
