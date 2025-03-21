import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message import ImgDetectionsExtended, Map2D


class ApplyColormap(dai.node.HostNode):
    """A host node that applies a colormap to a given 2D array (e.g. depth maps,
    segmentation masks, heatmaps, etc.).

    Attributes
    ----------
    colormap_value : Optional[int]
        OpenCV colormap enum value. Determines the applied color mapping.
    max_value : Optional[int]
        Maximum value to consider for normalization. If set lower than the map's actual maximum, the map's maximum will be used instead.
    arr : dai.ImgFrame or Map2D or ImgDetectionsExtended
        The input message with a 2D array.
    output : dai.ImgFrame
        The output message for a colorized frame.
    """

    def __init__(
        self, colormap_value: int = cv2.COLORMAP_HOT, max_value: int = 0
    ) -> None:
        super().__init__()

        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.setColormap(colormap_value)
        self.setMaxDisparity(max_value)

        self._platform = (
            self.getParentPipeline().getDefaultDevice().getPlatformAsString()
        )

    def setColormap(self, colormap_value: int) -> None:
        """Sets the applied color mapping.

        @param colormap_value: OpenCV colormap enum value (e.g. cv2.COLORMAP_HOT)
        @type colormap_value: int
        """
        assert isinstance(colormap_value, int)
        colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), colormap_value)
        colormap[0] = [0, 0, 0]  # Set zero values to black
        self._colormap = colormap

    def setMaxDisparity(self, max_value: int) -> None:
        """Sets the maximum frame value for normalization.

        @param max_value: Maximum frame value.
        @type max_value: int
        """
        assert isinstance(max_value, int)
        self._max_value = max_value

    def build(self, arr: dai.Node.Output) -> "ApplyColormap":
        """Configures the node connections.

        @param frame: Output with 2D array.
        @type frame: depthai.Node.Output
        @return: The node object with input stream connected
        @rtype: ApplyColormap
        """
        self.link_args(arr)
        return self

    def process(self, msg: dai.Buffer) -> None:
        """Processes incoming 2D arrays and converts them to colored frames.

        @param msg: The input message with a 2D array.
        @type msg: dai.ImgFrame or Map2D or ImgDetectionsExtended
        """

        if isinstance(msg, dai.ImgFrame):
            arr = msg.getFrame()
        elif isinstance(msg, Map2D):
            arr = msg.map  # density map
        elif isinstance(msg, ImgDetectionsExtended):
            labels = {
                idx: detection.label for idx, detection in enumerate(msg.detections)
            }
            labels[-1] = -1  # background class
            arr = np.vectorize(lambda x: labels.get(x, -1))(
                msg.masks  # instance_segmentation_mask
            )  # semantic segmentation mask
        else:
            raise ValueError("Unsupported message type. Cannot obtain a 2D array.")

        # make sure that min value == 0 to ensure proper normalization
        arr += np.abs(arr.min()) if arr.min() < 0 else 0

        max_value = max(self._max_value, arr.max())
        if max_value == 0:
            color_arr = np.zeros(
                (arr.shape[0], arr.shape[1], 3),
                dtype=np.uint8,
            )
        else:
            color_arr = cv2.applyColorMap(
                ((arr / max_value) * 255).astype(np.uint8),
                self._colormap,
            )

        frame = dai.ImgFrame()
        frame.setCvFrame(
            color_arr,
            (
                dai.ImgFrame.Type.BGR888p
                if self._platform == "RVC2"
                else dai.ImgFrame.Type.BGR888i
            ),
        )
        frame.setTimestamp(frame.getTimestamp())
        frame.setSequenceNum(frame.getSequenceNum())

        self.out.send(frame)
