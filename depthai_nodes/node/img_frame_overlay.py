import cv2
import depthai as dai


class ImgFrameOverlay(dai.node.HostNode):
    """A host node that receives two dai.ImgFrame objects and overlays them into a single one.

    Attributes
    ----------
    frame1 : dai.ImgFrame
        The input message for the background frame.
    frame2 : dai.ImgFrame
        The input message for the foreground frame.
    alpha: float
        The weight of the background frame in the overlay. By default, the weight is 0.5
            which means that both frames are represented equally in the overlay.
    out : dai.ImgFrame
        The output message for the overlay frame.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.SetAlpha(alpha)

        self._platform = (
            self.getParentPipeline().getDefaultDevice().getPlatformAsString()
        )

    def SetAlpha(self, alpha: float) -> None:
        """Sets the alpha.

        @param alpha: The weight of the background frame in the overlay.
        @type alpha: float
        """
        if not isinstance(alpha, float):
            raise ValueError("Alpha must be a float")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self._alpha = alpha

    def build(
        self, frame1: dai.Node.Output, frame2: dai.Node.Output, alpha: float = None
    ) -> "ImgFrameOverlay":
        """Configures the node connections.

        @param frame1: The input message for the background frame.
        @type frame1: dai.Node.Output
        @param frame2: The input message for the foreground frame.
        @type frame2: dai.Node.Output
        @param alpha: The weight of the background frame in the overlay.
        @type alpha: float
        @return: The node object with the background and foreground streams overlaid.
        @rtype: ImgFrameOverlay
        """
        self.link_args(frame1, frame2)

        if alpha is not None:
            self.SetAlpha(alpha)

        return self

    def process(self, frame1: dai.Buffer, frame2: dai.Buffer) -> None:
        """Processes incoming frames and overlays them.

        @param frame1: The input message for the background frame.
        @type frame1: dai.ImgFrame
        @param frame2: The input message for the foreground frame.
        @type frame2: dai.ImgFrame
        """
        assert isinstance(frame1, dai.ImgFrame)
        assert isinstance(frame2, dai.ImgFrame)

        background_frame = frame1.getCvFrame()
        foreground_frame = frame2.getCvFrame()

        # reshape foreground to match the background shape
        foreground_frame = cv2.resize(
            foreground_frame,
            dsize=(background_frame.shape[1], background_frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        overlay_frame = cv2.addWeighted(
            background_frame,
            self._alpha,
            foreground_frame,
            1 - self._alpha,
            0,
        )

        overlay = dai.ImgFrame()
        overlay.setCvFrame(
            overlay_frame,
            (
                dai.ImgFrame.Type.BGR888i
                if self._platform == "RVC4"
                else dai.ImgFrame.Type.BGR888p
            ),
        )
        overlay.setTimestamp(frame1.getTimestamp())
        overlay.setSequenceNum(frame1.getSequenceNum())

        self.out.send(overlay)
