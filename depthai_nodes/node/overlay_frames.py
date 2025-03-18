import cv2
import depthai as dai


class OverlayFrames(dai.node.HostNode):
    """A host node that receives two dai.ImgFrame objects and overlays them into a
    single dai.ImgFrame object.

    Attributes
    ----------
    background_weight: float
        The weight of the background frame in the overlay frame.
    background : dai.ImgFrame
        The input message for the background frame.
    foreground : dai.ImgFrame
        The input message for the foreground frame.
    output : dai.ImgFrame
        The output message for the overlay frame.
    """

    def __init__(self, background_weight: float = 0.5) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

        self.SetBackgroundWeight(background_weight)

        self._platform = (
            self.getParentPipeline().getDefaultDevice().getPlatformAsString()
        )

    def SetBackgroundWeight(self, background_weight: float) -> None:
        """Sets the weight of the background frame in the overlay.

        @param background_weight: The weight of the background frame in the overlay.
        @type background_weight: float
        """
        assert isinstance(background_weight, float)
        assert 0.0 <= background_weight <= 1.0
        self._background_weight = background_weight

    def build(
        self, background: dai.Node.Output, foreground: dai.Node.Output
    ) -> "OverlayFrames":
        """Configures the node connections.

        @param background: The input message for the background frame.
        @type background: dai.Node.Output
        @param foreground: The input message for the foreground frame.
        @type foreground: dai.Node.Output
        @return: The node object with the background and foreground streams overlaid.
        @rtype: OverlayFrames
        """
        self.link_args(background, foreground)
        return self

    def process(self, background: dai.Buffer, foreground: dai.Buffer) -> None:
        """Processes incoming background and foreground frames and overlays them.

        @param background: The input message for the background frame.
        @type background: dai.ImgFrame
        @param foreground: The input message for the foreground frame.
        @type foreground: dai.ImgFrame
        """
        assert isinstance(background, dai.ImgFrame)
        assert isinstance(foreground, dai.ImgFrame)

        background_frame = background.getCvFrame()
        foreground_frame = foreground.getCvFrame()

        # reshape foreground to match the background shape
        foreground_frame = cv2.resize(
            foreground_frame,
            dsize=(background_frame.shape[1], background_frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        overlay_frame = cv2.addWeighted(
            background_frame,
            self._background_weight,
            foreground_frame,
            1 - self._background_weight,
            0,
        )

        overlay = dai.ImgFrame()
        overlay.setCvFrame(
            overlay_frame,
            (
                dai.ImgFrame.Type.BGR888p
                if self._platform == "RVC2"
                else dai.ImgFrame.Type.BGR888i
            ),
        )
        overlay.setTimestamp(background.getTimestamp())
        overlay.setSequenceNum(background.getSequenceNum())

        self.output.send(overlay)
