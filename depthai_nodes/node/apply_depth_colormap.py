from typing import Tuple, Union

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.node.base_host_node import BaseHostNode


class ApplyDepthColormap(BaseHostNode):
    """A host node that applies a colormap to a depth map using percentile-based
    normalization to reduce flicker.

    Works with RAW 2D dai.ImgFrame outputs such as stereo.depth and stereo.disparity frames.
    Percentile normalization is typically more beneficial for stereo.depth since disparity often has a fixed output range.

    Invalid depth values (<= 0) are ignored when computing percentiles and are rendered as black in the output.

    Parameters
    ----------
    colormap_value : Union[int, np.ndarray], optional
        OpenCV colormap enum (e.g. cv2.COLORMAP_JET) or a custom OpenCV-compatible
        colormap LUT. Default is cv2.COLORMAP_JET.
    p_low : float, optional
        Lower normalization percentile in [0, 100). Default 2.0.
    p_high : float, optional
        Upper normalization percentile in (0, 100]. Default 98.0.

    Inputs
    ------
    frame : dai.ImgFrame
        Input message containing a 2D array to be colorized.

    Outputs
    -------
    output : dai.ImgFrame
        Colorized output frame (3-channel BGR).
    """

    def __init__(
        self,
        colormap_value: Union[int, np.ndarray] = cv2.COLORMAP_JET,
        p_low: float = 2.0,
        p_high: float = 98.0,
    ) -> None:
        super().__init__()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self._colormap = self._make_colormap(colormap_value)
        self._p_low, self._p_high = self._validate_percentile_range(p_low, p_high)

        self._logger.debug(
            "ApplyDepthColormap initialized with colormap_value=%s, p_low=%s, p_high=%s",
            colormap_value,
            self._p_low,
            self._p_high,
        )

    def setColormap(self, colormap_value: Union[int, np.ndarray]) -> None:
        """Sets the applied color mapping.

        @param colormap_value: OpenCV colormap enum value (e.g. cv2.COLORMAP_HOT) or a
            custom, OpenCV compatible, colormap definition
        @type colormap_value: Union[int, np.ndarray]
        """
        self._colormap = self._make_colormap(colormap_value)
        if isinstance(colormap_value, int):
            self._logger.debug("Colormap set to OpenCV enum: %s", colormap_value)
        else:
            self._logger.debug("Colormap set to custom LUT")

    def setPercentileRange(self, low: float, high: float) -> None:
        """Set the percentile clipping range used when normalization is PERCENTILE.

        @param low: Lower percentile in [0, 100).
        @param high: Higher percentile in (0, 100].
        """
        self._p_low, self._p_high = self._validate_percentile_range(low, high)
        self._logger.debug(
            "Percentile range set to low=%s, high=%s", self._p_low, self._p_high
        )

    def build(self, frame: dai.Node.Output) -> "ApplyDepthColormap":
        """Configures the node connections.

        @param frame: Node output that produces a RAW depth dai.ImgFrame.
        @type frame: dai.Node.Output
        @return: The configured node instance.
        @rtype: ApplyDepthColormap

        @example:
            >>> depth_color = ApplyDepthColormap(p_low=2, p_high=98).build(depth_out)
            >>> depth_color.setColormap(cv2.COLORMAP_JET)
        """
        self.link_args(frame)
        self._logger.debug("ApplyDepthColormap built")
        return self

    def process(self, frame: dai.Buffer) -> None:
        self._logger.debug("Processing new input")
        depth = self._get_depth_map(frame)

        invalid_depth_mask = depth <= 0
        valid = depth[~invalid_depth_mask]
        if valid.size == 0:
            color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
            self.out.send(self._build_output_frame(color, frame))
            return

        low, high = self._compute_normalization_bounds(valid)
        if high <= low or low <= 0:
            color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
            self.out.send(self._build_output_frame(color, frame))
            return

        color = self._colorize(depth, invalid_depth_mask, low, high)

        out = self._build_output_frame(color, frame)
        self._logger.debug("ImgFrame message created")

        self.out.send(out)
        self._logger.debug("Message sent successfully")

    def _validate_percentile_range(
        self, low: float, high: float
    ) -> Tuple[float, float]:
        low = float(low)
        high = float(high)
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("Percentile range must satisfy 0 <= low < high <= 100.")
        return low, high

    def _make_colormap(self, colormap_value: Union[int, np.ndarray]) -> np.ndarray:
        if isinstance(colormap_value, int):
            colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), colormap_value)
            return colormap

        if (
            isinstance(colormap_value, np.ndarray)
            and colormap_value.shape == (256, 1, 3)
            and colormap_value.dtype == np.uint8
        ):
            return colormap_value

        raise ValueError(
            "colormap_value must be an integer or an OpenCV compatible colormap definition."
        )

    def _get_depth_map(self, msg: dai.Buffer) -> np.ndarray:
        if not isinstance(msg, dai.ImgFrame):
            raise TypeError(
                f"Unsupported input type {type(msg)}, expected dai.ImgFrame."
            )
        if not msg.getType().name.startswith("RAW"):
            raise TypeError(f"Expected image type RAW, got {msg.getType().name}")
        return msg.getCvFrame()

    def _compute_normalization_bounds(
        self, valid_depth_values: np.ndarray
    ) -> Tuple[float, float]:
        low = float(np.percentile(valid_depth_values, self._p_low))
        high = float(np.percentile(valid_depth_values, self._p_high))
        return low, high

    def _colorize(
        self,
        depth: np.ndarray,
        invalid_depth_mask: np.ndarray,
        min_value: float,
        max_value: float,
    ) -> np.ndarray:
        depth = depth.astype(np.float32, copy=False)
        scaled = (depth - min_value) / (max_value - min_value) * 255.0
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)
        scaled[invalid_depth_mask] = 0  # invalid -> 0 so it becomes black

        color = cv2.applyColorMap(scaled, self._colormap)
        color[invalid_depth_mask] = 0
        return color

    def _build_output_frame(
        self, color_map: np.ndarray, src_frame: dai.Buffer
    ) -> dai.ImgFrame:
        frame = dai.ImgFrame()
        frame.setCvFrame(color_map, self._img_frame_type)
        frame.setTimestamp(src_frame.getTimestamp())
        frame.setSequenceNum(src_frame.getSequenceNum())
        frame.setTimestampDevice(src_frame.getTimestampDevice())

        t = src_frame.getTransformation()
        if t is not None:
            frame.setTransformation(t)

        return frame
