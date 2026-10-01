import cv2
import depthai as dai

from depthai_nodes.node.base_host_node import BaseHostNode


class ImgFrameOverlay(BaseHostNode):
    """Blend two image streams into a single frame.

    Args:
        alpha: Background weight between 0 and 1. The default of 0.5 gives background
            and foreground equal weight.
        preserveBackground: Preserve background pixels where the foreground is zero.

    Attributes:
        out: Output stream of blended ``dai.ImgFrame`` messages.
    """

    def __init__(self, alpha: float = 0.5, preserveBackground: bool = False) -> None:
        super().__init__()
        self.setAlpha(alpha)
        self.setPreserveBackground(preserveBackground)
        self._logger.debug(
            f"ImgFrameOverlay initialized with alpha={alpha}, preserve_background={preserveBackground}"
        )

    def setAlpha(self, alpha: float) -> None:
        """Set the background contribution used during overlay.

        Args:
            alpha: Weight of the background frame in the blended output.
        """
        if not isinstance(alpha, float):
            raise ValueError("Alpha must be a float")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self._alpha = alpha
        self._logger.debug(f"Alpha set to {self._alpha}")

    def setPreserveBackground(self, preserveBackground: bool) -> None:
        """Set whether zero-valued foreground pixels preserve the background.

        Args:
            preserveBackground: If True, zero areas in the foreground frame are ignored
                in the output image.
        """
        if not isinstance(preserveBackground, bool):
            raise ValueError("preserveBackground must be a boolean")
        self._preserve_background = preserveBackground

    def build(
        self,
        frame1: dai.Node.Output,
        frame2: dai.Node.Output,
        alpha: float | None = None,
        preserveBackground: bool | None = None,
    ) -> "ImgFrameOverlay":
        """Connect the input streams and optionally update overlay settings.

        Args:
            frame1: Upstream output producing the background frame.
            frame2: Upstream output producing the foreground frame.
            alpha: Optional blend weight for the background frame.
            preserveBackground: Optional override for whether zero-valued foreground
                pixels preserve the background frame.

        Returns:
            The configured node instance.
        """
        self.link_args(frame1, frame2)

        if alpha is not None:
            self.setAlpha(alpha)
        if preserveBackground is not None:
            self.setPreserveBackground(preserveBackground)

        self._logger.debug(
            f"ImgFrameOverlay built with alpha={alpha}, preserve_background={preserveBackground}"
        )

        return self

    def process(self, frame1: dai.Buffer, frame2: dai.Buffer) -> None:
        """Overlay the foreground frame onto the background frame."""
        self._logger.debug("Processing new input")
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

        if self._preserve_background and foreground_frame is not None:
            # Ensure foreground is 3-channel RGB
            if len(foreground_frame.shape) == 2:  # grayscale
                foreground_rgb = cv2.cvtColor(
                    foreground_frame.astype("uint8"), cv2.COLOR_GRAY2BGR
                )
                mask = foreground_frame > 0
            else:  # already RGB
                foreground_rgb = foreground_frame
                mask = foreground_frame.max(axis=2) > 0

            if mask.any():
                overlay_frame = background_frame.copy()
                overlay_frame[mask] = cv2.addWeighted(
                    background_frame[mask],
                    self._alpha,
                    foreground_rgb[mask],
                    1 - self._alpha,
                    0,
                )
            else:
                overlay_frame = background_frame.copy()
        else:
            overlay_frame = cv2.addWeighted(
                background_frame,
                self._alpha,
                foreground_frame if foreground_frame is not None else background_frame,
                1 - self._alpha,
                0,
            )

        overlay = dai.ImgFrame()
        overlay.setCvFrame(
            overlay_frame,
            self._img_frame_type,
        )
        overlay.setTimestamp(frame1.getTimestamp())
        overlay.setSequenceNum(frame1.getSequenceNum())
        overlay.setTimestampDevice(frame1.getTimestampDevice())
        transformation = frame1.getTransformation()
        if transformation is not None:
            overlay.setTransformation(transformation)

        self._logger.debug("ImgFrame message created")

        self.out.send(overlay)

        self._logger.debug("Message sent successfully")
