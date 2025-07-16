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
    depth_alignment_socket : dai.CameraBoardSocket
        The camera socket used for depth alignment.
    delta : int
        The delta value for ROI calculation. Default is 5 - means 10x10 depth pixels around point for depth averaging.
    thresh_low : int
        The lower threshold for depth values. Default is 200 - means 20cm.
    thresh_high : int
        The upper threshold for depth values. Default is 30000 - means 30m.
    """

    # We need device object to get calibration data
    def __init__(
        self,
        calib_data: dai.CalibrationHandler,
        depth_alignment_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
        delta: int = 5,
        thresh_low: int = 200,
        thresh_high: int = 30000,
    ):
        self.calibData = calib_data
        self.depth_alignment_socket = depth_alignment_socket

        self.delta = delta
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

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
        self.thresh_low = threshold_low

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
        self.thresh_high = threshold_high

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
        self.delta = delta

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
        x = min(max(roi[0], self.delta), frame.shape[1] - self.delta)
        y = min(max(roi[1], self.delta), frame.shape[0] - self.delta)
        return [x - self.delta, y - self.delta, x + self.delta, y + self.delta]

    # roi has to be list of ints
    def calc_spatials(
        self,
        depthData: dai.ImgFrame,
        roi: List[int],
        averaging_method: Callable = np.mean,
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
        inRange = (self.thresh_low <= depthROI) & (depthROI <= self.thresh_high)

        valid_depths = depthROI[inRange]
        if valid_depths.size == 0:
            return {
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
            }
        else:
            averageDepth = averaging_method(valid_depths)

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
