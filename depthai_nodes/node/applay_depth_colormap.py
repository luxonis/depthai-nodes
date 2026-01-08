from typing import Union, Tuple, Optional

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.node.base_host_node import BaseHostNode


class ApplyDepthColormap(BaseHostNode):
    """
    A host node that applies a colormap to a depth map using percentile-based normalization to reduce flicker.

    Invalid depth values (<= 0) are ignored when computing percentiles and are rendered as black in the output.

    Attributes
    ----------
    colormap_value : Union[int, np.ndarray]
        OpenCV colormap enum value or a custom, OpenCV compatible, colormap. Determines the applied color mapping.
    p_low : float
        Lower normalization percentile in [0, 100). Default 2.0.
    p_high : float
        Upper normalization percentile in (0, 100]. Default 98.0.
    min_depth_mm : Optional[float]
        Optional fixed minimum depth bound in millimeters. If set, overrides the percentile-derived low bound.
    max_depth_mm : Optional[float]
        Optional fixed maximum depth bound in millimeters. If set, overrides the percentile-derived high bound.
    frame : dai.ImgFrame
        The input message with a 2D array.
    output : dai.ImgFrame
        The output message for a colorized frame.
    """
    def __init__(
        self,
        colormap_value: Union[int, np.ndarray] = cv2.COLORMAP_JET,
        p_low: float = 2.0,
        p_high: float = 98.0,
        min_depth_mm: Optional[float] = None,
        max_depth_mm: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self._min_depth_mm = None if min_depth_mm is None else float(min_depth_mm)
        self._max_depth_mm = None if max_depth_mm is None else float(max_depth_mm)

        if self._min_depth_mm is not None and self._max_depth_mm is not None:
            if self._max_depth_mm <= self._min_depth_mm:
                raise ValueError(
                    f"max_depth_mm must be > min_depth_mm. Got min_depth_mm={self._min_depth_mm}, "
                    f"max_depth_mm={self._max_depth_mm}."
                )

        self.setPercentileRange(low=p_low, high=p_high)
        self.setColormap(colormap_value)

    def build(self, frame: dai.Node.Output) -> "ApplyDepthColormap":
        """
        Configures the node connections.

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

    def process(self, msg: dai.Buffer) -> None:
        depth = self._get_depth_map(msg)

        invalid_depth_mask = depth <= 0
        valid = depth[~invalid_depth_mask]
        if valid.size == 0:
            color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
            self.out.send(self._build_output_frame(color, msg))
            return

        low, high = self._compute_normalization_bounds(valid)
        if high <= low or low <= 0:
            color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
            self.out.send(self._build_output_frame(color, msg))
            return

        color = self._colorize(depth, invalid_depth_mask, low, high)
        self.out.send(self._build_output_frame(color, msg))

    def setColormap(self, colormap_value: Union[int, np.ndarray]) -> None:
        """Sets the applied color mapping.

        @param colormap_value: OpenCV colormap enum value (e.g. cv2.COLORMAP_HOT) or a
            custom, OpenCV compatible, colormap definition
        @type colormap_value: Union[int, np.ndarray]
        """
        if isinstance(colormap_value, int):
            colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), colormap_value)
            colormap[0] = [0, 0, 0]  # Set zero values to black
            self._colormap = colormap
            self._logger.debug(f"Colormap set to {colormap_value}")
        elif (
            isinstance(colormap_value, np.ndarray)
            and colormap_value.shape == (256, 1, 3)
            and colormap_value.dtype == np.uint8
        ):
            self._colormap = colormap_value
            self._logger.debug("Colormap set to custom")
        else:
            raise ValueError(
                "colormap_value must be an integer or an OpenCV compatible colormap definition."
            )

    def setPercentileRange(self, low: float, high: float) -> None:
        """
        Set the percentile clipping range used when normalization is PERCENTILE.

        @param low: Lower percentile in [0, 100).
        @param high: Higher percentile in (0, 100].
        """
        low = float(low)
        high = float(high)
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("Percentile range must satisfy 0 <= low < high <= 100.")
        self._p_low = low
        self._p_high = high
        self._logger.debug(
            f"Percentile range set to low={self._p_low}, high={self._p_high}"
        )

    def _get_depth_map(self, msg: dai.Buffer) -> np.ndarray:
        if not isinstance(msg, dai.ImgFrame):
            raise TypeError(f"Unsupported input type {type(msg)}, expected dai.ImgFrame.")
        if not msg.getType().name.startswith("RAW"):
            raise TypeError(f"Expected image type RAW, got {msg.getType().name}")
        return msg.getCvFrame()

    def _compute_normalization_bounds(self, valid_depth_values: np.ndarray) -> Tuple[float, float]:
        low = float(np.percentile(valid_depth_values, self._p_low))
        high = float(np.percentile(valid_depth_values, self._p_high))

        if self._min_depth_mm is not None:
            low = self._min_depth_mm
        if self._max_depth_mm is not None:
            high = self._max_depth_mm

        return low, high

    def _colorize(
        self,
        depth: np.ndarray,
        invalid_depth_mask: np.ndarray,
        min_value: float,
        max_value: float,
    ) -> np.ndarray:
        depth = depth.astype(np.float32, copy=False)
        depth = np.clip(depth, min_value, max_value)
        scaled = ((depth - min_value) / (max_value - min_value) * 255.0)
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)
        scaled[invalid_depth_mask] = 0  # invalid -> 0 so it becomes black

        color = cv2.applyColorMap(scaled, self._colormap)
        color[invalid_depth_mask] = 0
        return color

    def _build_output_frame(self, color_map: np.ndarray, src_frame: dai.Buffer) -> dai.ImgFrame:
        frame = dai.ImgFrame()
        frame.setCvFrame(color_map, self._img_frame_type)

        frame.setTimestamp(src_frame.getTimestamp())
        frame.setSequenceNum(src_frame.getSequenceNum())
        frame.setTimestampDevice(src_frame.getTimestampDevice())

        t = src_frame.getTransformation()
        if t is not None:
            frame.setTransformation(t)

        return frame
