from typing import Union

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message import ImgDetectionsExtended, Map2D, SegmentationMask
from depthai_nodes.message.utils import copy_message
from depthai_nodes.node.base_host_node import BaseHostNode


class ApplyColormap(BaseHostNode):
    """A host node that applies a colormap to a given 2D array (e.g. depth maps,
    segmentation masks, heatmaps, etc.).

    Attributes
    ----------
    colormap_value : Union[int, np.ndarray]
        OpenCV colormap enum value or a custom, OpenCV compatible, colormap. Determines the applied color mapping.
    max_value : int
        Maximum value to consider for normalization. If set lower than the map's actual maximum, the map's maximum will be used instead.
    instance_to_semantic_mask : bool
        If True, converts instance segmentation masks to semantic segmentation masks. Note that this is only relevant for ImgDetectionsExtended messages.
    normalization : str
        Normalization mode. "max" (default) uses maximum value; "percentile" uses percentile clipping (useful for depth maps).
    percentile_low : float
        Lower percentile [0-100] for clipping when normalization="percentile".
    percentile_high : float
        Higher percentile [0-100] for clipping when normalization="percentile".
    ignore_zero_in_percentile : bool
        If True, 0 values are ignored when calculating percentiles (useful for depth).
    arr : dai.ImgFrame or Map2D or ImgDetectionsExtended
        The input message with a 2D array.
    output : dai.ImgFrame
        The output message for a colorized frame.
    """

    def __init__(
        self,
        colormap_value: Union[int, np.ndarray] = cv2.COLORMAP_JET,
        max_value: int = 0,
        instance_to_semantic_mask: bool = False,
    ) -> None:
        super().__init__()

        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self._normalization: str = "max"  # "max" | "percentile"
        self._percentile_low: float = 2.0
        self._percentile_high: float = 98.0
        self._ignore_zero_in_percentile: bool = False

        self.setColormap(colormap_value)
        self.setMaxValue(max_value)
        self.setInstanceToSemanticMask(instance_to_semantic_mask)

        self._logger.debug(
            f"ApplyColormap initialized with colormap_value={colormap_value}, max_value={max_value}, instance_to_semantic_mask={instance_to_semantic_mask}",
        )

    def setDepthPreset(self) -> None:
        """Enables depth-friendly normalization to reduce flickering.

        Uses percentile clipping (2-98%) and ignores invalid depth (0).
        """
        self._normalization = "percentile"
        self._percentile_low = 2.0
        self._percentile_high = 98.0
        self._ignore_zero_in_percentile = True
        self._logger.debug("Depth preset enabled")

    def setNormalization(self, normalization: str) -> None:
        """Sets the normalization mode.

        @param normalization: "max" | "percentile".
        @type normalization: str
        """
        normalization = str(normalization).lower().strip()
        if normalization not in ("max", "percentile"):
            raise ValueError("normalization must be one of: 'max', 'percentile'.")
        self._normalization = normalization
        self._logger.debug(f"Normalization set to {self._normalization}")

    def setPercentileRange(self, low: float, high: float) -> None:
        """Sets percentile range used when normalization='percentile'.

        @param low: Lower percentile in [0, 100).
        @param high: Higher percentile in (0, 100].
        """
        low = float(low)
        high = float(high)
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("Percentile range must satisfy 0 <= low < high <= 100.")
        self._percentile_low = low
        self._percentile_high = high
        self._logger.debug(
            f"Percentile range set to low={self._percentile_low}, high={self._percentile_high}"
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
            self._logger.debug(f"Colormap set to {self._colormap}")
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

    def setInstanceToSemanticMask(self, instance_to_semantic_mask: bool) -> None:
        """Sets the instance to semantic mask flag.

        @param instance_to_semantic_mask: If True, converts instance segmentation masks
            to semantic segmentation masks.
        @type instance_to_semantic_mask: bool
        """
        if not isinstance(instance_to_semantic_mask, bool):
            raise ValueError("instance_to_semantic_mask must be a boolean.")
        self._instance_to_semantic_mask = instance_to_semantic_mask
        self._logger.debug(
            f"Instance to semantic mask set to {self._instance_to_semantic_mask}"
        )

    def build(self, arr: dai.Node.Output) -> "ApplyColormap":
        """Configures the node connections.

        @param frame: Output with 2D array.
        @type frame: depthai.Node.Output
        @return: The node object with input stream connected
        @rtype: ApplyColormap
        """
        self.link_args(arr)
        self._logger.debug("ApplyColormap built")
        return self

    def process(self, msg: dai.Buffer) -> None:
        """Processes incoming 2D arrays and converts them to colored frames.

        @param msg: The input message with a 2D array.
        @type msg: dai.ImgFrame or Map2D or ImgDetectionsExtended
        @param instance_to_semantic_segmentation: If True, converts instance
            segmentation masks to semantic segmentation masks.
        @type instance_to_semantic_segmentation: bool
        """
        self._logger.debug("Processing new input")
        msg_copy = copy_message(msg)

        if isinstance(msg, SegmentationMask):
            arr = msg_copy.mask
        elif isinstance(msg, dai.ImgFrame):
            if not msg.getType().name.startswith("RAW"):
                raise TypeError(f"Expected image type RAW, got {msg.getType().name}")
            arr = msg.getCvFrame()
        elif isinstance(msg, Map2D):
            arr = msg_copy.map
        elif isinstance(msg, ImgDetectionsExtended):
            if self._instance_to_semantic_mask:
                labels = {
                    idx: detection.label for idx, detection in enumerate(msg.detections)
                }
                labels[-1] = -1  # background class
                arr = np.vectorize(lambda x: labels.get(x, -1))(
                    msg_copy.masks  # instance segmentation mask
                )  # semantic segmentation mask
            else:
                arr = msg_copy.masks  # semantic segmentation mask
        else:
            raise ValueError(f"Unsupported input type {type(msg)}.")

        # make sure that min value == 0 to ensure proper normalization
        if not self._ignore_zero_in_percentile:
            arr += np.abs(arr.min()) if arr.min() < 0 else 0

        # Calculate Normalization Bounds
        low_value, high_value = 0.0, 0.0

        if self._normalization == "max":
            high_value = float(max(self._max_value, int(arr.max())))

        elif self._normalization == "percentile":
            if self._ignore_zero_in_percentile:
                valid = arr[arr > 0]
            else:
                valid = arr.reshape(-1)

            if valid.size > 0:
                low_value = float(np.percentile(valid, self._percentile_low))
                high_value = float(np.percentile(valid, self._percentile_high))

                # If user set a manual max_value, ensure high_value is at least that
                if self._max_value > 0:
                    high_value = max(high_value, float(self._max_value))
        else:
            raise ValueError(f"Unsupported normalization mode {self._normalization}.")

        if high_value <= low_value:
            color_arr = np.zeros(
                (arr.shape[0], arr.shape[1], 3),
                dtype=np.uint8,
            )
        else:
            scaled = (
                (arr.astype(np.float32) - low_value) / (high_value - low_value)
            ) * 255.0
            scaled = np.clip(scaled, 0, 255).astype(np.uint8)
            color_arr = cv2.applyColorMap(scaled, self._colormap)

        frame = dai.ImgFrame()
        frame.setCvFrame(
            color_arr,
            self._img_frame_type,
        )

        frame.setTimestamp(msg.getTimestamp())
        frame.setSequenceNum(msg.getSequenceNum())
        frame.setTimestampDevice(msg.getTimestampDevice())
        transformation = msg.getTransformation()
        if transformation is not None:
            frame.setTransformation(transformation)

        self._logger.debug("ImgFrame message created")

        self.out.send(frame)

        self._logger.debug("Message sent successfully")
