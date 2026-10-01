from collections.abc import Callable

import depthai as dai
import numpy as np


class HostSpatialsCalc:
    """Compute camera-space coordinates from depth regions.

    Depth values and returned coordinates use the same units, normally millimeters.

    Note:
        Four-coordinate ROIs are used as NumPy slice bounds without clipping. Point
        inputs are clamped so the sampling square fits inside the frame.
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
        """Configure calibration, point sampling, and accepted depth range.

        Args:
            calibData: Device calibration used to obtain camera intrinsics.
            depthAlignmentSocket: Camera to which depth pixels are aligned.
            delta: Half-size in pixels of the square sampled around point inputs.
            threshLow: Inclusive minimum accepted depth, in the depth frame units.
            threshHigh: Inclusive maximum accepted depth, in the depth frame units.
        """
        self.calibData = calibData
        self.depth_alignment_socket = depthAlignmentSocket

        self.delta = delta
        self.thresh_low = threshLow
        self.thresh_high = threshHigh

    def setLowerThreshold(self, thresholdLow: int) -> None:
        """Set the lower depth threshold used during ROI averaging.

        Args:
            thresholdLow: Lower accepted depth value.
        """
        if not isinstance(thresholdLow, int):
            if isinstance(thresholdLow, float):
                thresholdLow = int(thresholdLow)
            else:
                raise TypeError(
                    f"Threshold has to be an integer or float! Got {type(thresholdLow)}"
                )
        self.thresh_low = thresholdLow

    def setUpperThreshold(self, thresholdHigh: int) -> None:
        """Set the upper depth threshold used during ROI averaging.

        Args:
            thresholdHigh: Upper accepted depth value.
        """
        if not isinstance(thresholdHigh, int):
            if isinstance(thresholdHigh, float):
                thresholdHigh = int(thresholdHigh)
            else:
                raise TypeError(
                    f"Threshold has to be an integer or float! Got {type(thresholdHigh)}"
                )
        self.thresh_high = thresholdHigh

    def setDeltaRoi(self, delta: int) -> None:
        """Set the point-sampling square half-size.

        Args:
            delta: Half-size in pixels. Floating-point values are truncated to integers.

        Raises:
            TypeError: If delta is neither an int nor a float.
        """
        if not isinstance(delta, int):
            if isinstance(delta, float):
                delta = int(delta)
            else:
                raise TypeError(
                    f"Delta has to be an integer or float! Got {type(delta)}"
                )
        self.delta = delta

    def calcSpatials(
        self,
        depthData: dai.ImgFrame,
        roi: list[int],
        averagingMethod: Callable = np.mean,
    ) -> dict[str, float]:
        """Project the depth-region centroid into camera space.

        Args:
            depthData: Depth frame aligned to the configured camera.
            roi: Pixel coordinates as ``[xmin, ymin, xmax, ymax]``, or a point ``[x,
                y]`` expanded by ``delta``. Slice upper bounds are exclusive.
            averagingMethod: Reducer for depth samples within the inclusive configured
                thresholds; defaults to the mean.

        Returns:
            Dictionary with ``x``, ``y``, and ``z`` in depth-frame units. All values are
            zero if no samples pass the depth thresholds.

        Raises:
            ValueError: If the ROI contains neither two nor four coordinates.
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
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
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

    def _check_input(self, roi: list[int], frame: np.ndarray) -> list[int]:
        if len(roi) == 4:
            return roi
        if len(roi) != 2:
            raise ValueError(
                "You have to pass either ROI (4 values) or point (2 values)!"
            )
        x = min(max(roi[0], self.delta), frame.shape[1] - self.delta)
        y = min(max(roi[1], self.delta), frame.shape[0] - self.delta)
        return [x - self.delta, y - self.delta, x + self.delta, y + self.delta]
