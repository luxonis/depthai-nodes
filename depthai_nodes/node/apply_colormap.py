import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message import Map2D


class ApplyColormap(dai.node.HostNode):
    """A host node that applies a colormap to a given 2D frame.

    Attributes
    ----------
    colormap_value : Optional[int]
        OpenCV colormap enum value. Determines the applied color mapping.
    max_value : Optional[int]
        Maximum value to consider for normalization. If set lower than the map's actual maximum, the map's maximum will be used instead.
    frame : Map2D or dai.ImgFrame
        The input message for a 2D frame.
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

    def build(self, frame: dai.Node.Output) -> "ApplyColormap":
        """Configures the node connections.

        @param frame: Output with 2D frame.
        @type frame: depthai.Node.Output
        @return: The node object with input stream connected
        @rtype: ApplyColormap
        """
        self.link_args(frame)
        return self

    def process(self, frame: dai.Buffer) -> None:
        """Processes incoming 2D frames and converts them to colored frames.

        @param frame: Input 2D frame.
        @type frame: Map2D or dai.ImgFrame
        """

        if isinstance(frame, dai.ImgFrame):
            raw = frame.getFrame()
        elif isinstance(frame, Map2D):
            raw = frame.map
        else:
            raise ValueError("Unsupported 2D frame type")

        max_value = max(self._max_value, raw.max())
        if max_value == 0:
            color = np.zeros(
                (raw.shape[0], raw.shape[1], 3),
                dtype=np.uint8,
            )
        else:
            color = cv2.applyColorMap(
                ((raw / max_value) * 255).astype(np.uint8),
                self._colormap,
            )

        frame_colorized = dai.ImgFrame()
        frame_colorized.setCvFrame(
            color,
            (
                dai.ImgFrame.Type.BGR888p
                if self._platform == "RVC2"
                else dai.ImgFrame.Type.BGR888i
            ),
        )
        frame_colorized.setTimestamp(frame.getTimestamp())
        frame_colorized.setSequenceNum(frame.getSequenceNum())

        self.out.send(frame_colorized)
