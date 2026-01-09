from typing import Union

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message import ImgDetectionsExtended, Map2D, SegmentationMask
from depthai_nodes.message.utils import copy_message
from depthai_nodes.node.base_host_node import BaseHostNode


class ApplyColormap(BaseHostNode):
    """A host node that applies a colormap to a 2D array (e.g. depth maps, segmentation
    masks, heatmaps, etc.).

    This node is generic and uses per-frame max-value normalization. For depth visualization prefer 'ApplyDepthColormap'
    to avoid flicker caused by the changing normalization range.

    Attributes
    ----------
    colormap_value : Union[int, np.ndarray]
        OpenCV colormap enum value or a custom, OpenCV compatible, colormap. Determines the applied color mapping.
    max_value : int
        Maximum value to consider for normalization. If set lower than the map's actual maximum, the map's maximum will be used instead.
    frame : dai.ImgFrame or Map2D or ImgDetectionsExtended or SegmentationMask
        The input message with a 2D array.
    output : dai.ImgFrame
        The output message for a colorized frame.
    """

    def __init__(
        self,
        colormap_value: Union[int, np.ndarray] = cv2.COLORMAP_JET,
        max_value: int = 0,
    ) -> None:
        super().__init__()

        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.setColormap(colormap_value)
        self.setMaxValue(max_value)

        self._logger.debug(
            f"ApplyColormap initialized with colormap_value={colormap_value}, max_value={max_value}",
        )

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

    def setMaxValue(self, max_value: int) -> None:
        """Sets the maximum frame value for normalization.

        @param max_value: Maximum frame value.
        @type max_value: int
        """
        if not isinstance(max_value, int):
            raise ValueError("max_value must be an integer.")
        self._max_value = max_value
        self._logger.debug(f"Max value set to {self._max_value}")

    def build(self, frame: dai.Node.Output) -> "ApplyColormap":
        """Configures the node connections.

        @param frame: Output with 2D array.
        @type frame: depthai.Node.Output
        @return: The node object with input stream connected
        @rtype: ApplyColormap
        """
        self.link_args(frame)
        self._logger.debug("ApplyColormap built")
        return self

    def process(self, frame: dai.Buffer) -> None:
        self._logger.debug("Processing new input")
        input_map = self._get_input_map(frame)

        color_map = self._colorize(input_map)

        out = self._build_output_frame(color_map, frame)
        self._logger.debug("ImgFrame message created")

        self.out.send(out)
        self._logger.debug("Message sent successfully")

    def _get_input_map(self, msg: dai.Buffer) -> np.ndarray:
        if isinstance(msg, dai.ImgFrame):
            if not msg.getType().name.startswith("RAW"):
                raise TypeError(f"Expected image type RAW, got {msg.getType().name}")
            return msg.getCvFrame()

        msg_copy = copy_message(msg)

        if isinstance(msg_copy, SegmentationMask):
            return msg_copy.mask

        if isinstance(msg_copy, Map2D):
            return msg_copy.map

        if isinstance(msg_copy, ImgDetectionsExtended):
            return msg_copy.masks

        raise ValueError(f"Unsupported input type {type(msg_copy)}.")

    def _colorize(self, input_map: np.ndarray) -> np.ndarray:
        # make sure that min value == 0 to ensure proper normalization
        if input_map.min() < 0:
            input_map = input_map + np.abs(input_map.min())

        max_value = max(self._max_value, input_map.max())
        if max_value == 0:
            color_map = np.zeros(
                (input_map.shape[0], input_map.shape[1], 3),
                dtype=np.uint8,
            )
        else:
            color_map = cv2.applyColorMap(
                ((input_map / max_value) * 255).astype(np.uint8),
                self._colormap,
            )

        return color_map

    def _build_output_frame(
        self, color_map: np.ndarray, src_frame: dai.Buffer
    ) -> dai.ImgFrame:
        frame = dai.ImgFrame()
        frame.setCvFrame(color_map, self._img_frame_type)

        frame.setTimestamp(src_frame.getTimestamp())
        frame.setSequenceNum(src_frame.getSequenceNum())
        frame.setTimestampDevice(src_frame.getTimestampDevice())

        transformation = src_frame.getTransformation()
        if transformation is not None:
            frame.setTransformation(transformation)

        self._logger.debug("ImgFrame message created")
        return frame
