# HostSpatialsCalc implementation taken from here:
# https://github.com/luxonis/depthai-experiments/blob/d10736715bef1663d984196f8528610a614e4b75/gen2-calc-spatials-on-host/calc.py

from typing import Dict, List

import depthai as dai
import numpy as np


class HostSpatialsCalc:
    """HostSpatialsCalc is a helper class for calculating spatial coordinates from depth
    data.

    Attributes
    ----------
    calibData : dai.CalibrationHandler
        Calibration data handler for the device.
    depth_alignment_socket : dai.CameraBoardSocket
        The camera socket used for depth alignment.
    DELTA : int
        The delta value for ROI calculation.
    THRESH_LOW : int
        The lower threshold for depth values.
    THRESH_HIGH : int
        The upper threshold for depth values.

        setLowerThreshold(threshold_low): Sets the lower threshold for depth values.
        setUpperThreshold(threshold_high): Sets the upper threshold for depth values.
        setDeltaRoi(delta): Sets the delta value for ROI calculation.
        _check_input(roi, frame): Checks if the input is ROI or point and converts point to ROI if necessary.
        calc_spatials(depthData, roi, averaging_method): Calculates spatial coordinates from depth data within the specified ROI.
    """

    # We need device object to get calibration data
    def __init__(
        self,
        calib_data: dai.CalibrationHandler,
        depth_alignment_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
    ):
        self.calibData = calib_data
        self.depth_alignment_socket = depth_alignment_socket

        # Values
        self.DELTA = 5  # Take 10x10 depth pixels around point for depth averaging
        self.THRESH_LOW = 200  # 20cm
        self.THRESH_HIGH = 30000  # 30m

    def setLowerThreshold(self, threshold_low: int) -> None:
        """Sets the lower threshold for depth values.

        @param threshold_low: The lower threshold for depth values.
        @type threshold_low: int
        """
        if not isinstance(threshold_low, int):
            if isinstance(threshold_low, float):
                threshold_low = int(threshold_low)
            else:
                raise TypeError(
                    "Threshold has to be an integer or float! Got {}".format(
                        type(threshold_low)
                    )
                )
        self.THRESH_LOW = threshold_low

    def setUpperThreshold(self, threshold_high: int) -> None:
        """Sets the upper threshold for depth values.

        @param threshold_high: The upper threshold for depth values.
        @type threshold_high: int
        """
        if not isinstance(threshold_high, int):
            if isinstance(threshold_high, float):
                threshold_high = int(threshold_high)
            else:
                raise TypeError(
                    "Threshold has to be an integer or float! Got {}".format(
                        type(threshold_high)
                    )
                )
        self.THRESH_HIGH = threshold_high

    def setDeltaRoi(self, delta: int) -> None:
        """Sets the delta value for ROI calculation.

        @param delta: The delta value for ROI calculation.
        @type delta: int
        """
        if not isinstance(delta, int):
            if isinstance(delta, float):
                delta = int(delta)
            else:
                raise TypeError(
                    "Delta has to be an integer or float! Got {}".format(type(delta))
                )
        self.DELTA = delta

    def _check_input(self, roi: List[int], frame: np.ndarray) -> List[int]:
        """Checks if the input is ROI or point and converts point to ROI if necessary.

        @param roi: The region of interest (ROI) or point.
        @type roi: List[int]
        @param frame: The depth frame.
        @type frame: np.ndarray
        @return: The region of interest (ROI).
        @rtype: List[int]
        """
        if len(roi) == 4:
            return roi
        if len(roi) != 2:
            raise ValueError(
                "You have to pass either ROI (4 values) or point (2 values)!"
            )
        # Limit the point so ROI won't be outside the frame
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x - self.DELTA, y - self.DELTA, x + self.DELTA, y + self.DELTA)

    # roi has to be list of ints
    def calc_spatials(
        self,
        depthData: dai.ImgFrame,
        roi: List[int],
        averaging_method: callable = np.mean,
    ) -> Dict[str, float]:
        """Calculates spatial coordinates from depth data within the specified ROI.

        @param depthData: The depth data.
        @type depthData: dai.ImgFrame
        @param roi: The region of interest (ROI) or point.
        @type roi: List[int]
        @param averaging_method: The method for averaging the depth values.
        @type averaging_method: callable
        @return: The spatial coordinates.
        @rtype: Dict[str, float]
        """
        depthFrame = depthData.getFrame()

        roi = self._check_input(
            roi, depthFrame
        )  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])

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
