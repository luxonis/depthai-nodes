import cv2
import depthai as dai
import numpy as np


class DepthColorTransform(dai.node.HostNode):
    """Postprocessing node for colorizing disparity/depth frames.

    Converts grayscale disparity/depth frames into color-mapped visualization using OpenCV colormaps.

    Output Message/s
    ----------------
    **Type** : ImgFrame(dai.Buffer)

    **Description**: Colorized BGR888i frame where depth/disparity values are mapped to colors using the specified colormap.
    """

    def __init__(self) -> None:
        """Initializes the depth colorization node with default HOT colormap."""
        super().__init__()
        # TODO: Cannot access attribute "setPossibleDatatypes" for class "Output"

        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        self._max_disparity = 0
        self.setColormap(cv2.COLORMAP_HOT)

    def setColormap(self, colormap_value: int) -> None:
        """Sets the colormap used for depth visualization.

        @param colormap_value: OpenCV colormap enum value (e.g. cv2.COLORMAP_HOT)
        @type colormap_value: int
        """
        color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), colormap_value)
        color_map[0] = [0, 0, 0]  # Set zero values to black
        self._colormap = color_map

    def build(self, disparity_frames: dai.Node.Output) -> "DepthColorTransform":
        """Configures the node connections.

        @param disparity_frames: Output with disparity/depth frames
        @type disparity_frames: depthai.Node.Output
        @return: The node object with input stream connected
        @rtype: DepthColorTransform
        """
        self.link_args(disparity_frames)
        return self

    def setMaxDisparity(self, max_disparity: int) -> None:
        """Sets the maximum disparity value for normalization.

        @param max_disparity: Maximum disparity value. If not set, uses frame's maximum
            value. If set smaller than maximum of a frame, maximum of the frame will be
            used instead.
        @type max_disparity: int
        """
        self._max_disparity = max_disparity

    def process(self, disparity_frame: dai.Buffer) -> None:
        """Processes incoming disparity frames and converts them to colored frames.

        @param disparity_frame: Input disparity/depth frame
        @type disparity_frame: depthai.ImgFrame
        """
        assert isinstance(disparity_frame, dai.ImgFrame)

        frame = disparity_frame.getFrame()
        maxDisparity = max(self._max_disparity, frame.max())
        if maxDisparity == 0:
            colorizedDisparity = np.zeros(
                (frame.shape[0], frame.shape[1], 3), dtype=np.uint8
            )
        else:
            colorizedDisparity = cv2.applyColorMap(
                ((frame / maxDisparity) * 255).astype(np.uint8), self._colormap
            )
        resultFrame = dai.ImgFrame()
        resultFrame.setCvFrame(colorizedDisparity, dai.ImgFrame.Type.BGR888i)
        resultFrame.setTimestamp(disparity_frame.getTimestamp())
        self.out.send(resultFrame)
