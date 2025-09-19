from typing import Optional

import cv2
import depthai as dai

from depthai_nodes.node.base_host_node import BaseHostNode


class ImgFrameOverlay(BaseHostNode):
    """A host node that receives two dai.ImgFrame objects and overlays them into a
    single one.

    Attributes
    ----------
    frame1 : dai.ImgFrame
        The input message for the background frame.
    frame2 : dai.ImgFrame
        The input message for the foreground frame.
    alpha: float
        The weight of the background frame in the overlay. By default, the weight is 0.5
            which means that both frames are represented equally in the overlay.
    preserve_background: bool
        If True, zero areas in the foreground frame are ignored in the output overlay frame. Default is False.
    out : dai.ImgFrame
        The output message for the overlay frame.
    """

    def __init__(self, alpha: float = 0.5, preserve_background: bool = False) -> None:
        super().__init__()
        self.setAlpha(alpha)
        self.setPreserveBackground(preserve_background)
        self._logger.debug(
            f"ImgFrameOverlay initialized with alpha={alpha}, preserve_background={preserve_background}"
        )

    def setAlpha(self, alpha: float) -> None:
        """Sets the alpha.

        @param alpha: The weight of the background frame in the overlay.
        @type alpha: float
        """
        if not isinstance(alpha, float):
            raise ValueError("Alpha must be a float")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self._alpha = alpha
        self._logger.debug(f"Alpha set to {self._alpha}")

    def setPreserveBackground(self, preserve_background: bool) -> None:
        """Sets the preserve_background flag.

        @param preserve_background: If True, zero areas in the foreground frame are
            ignored in the output overlay frame.
        @type preserve_background: bool
        """
        if not isinstance(preserve_background, bool):
            raise ValueError("preserve_background must be a boolean")
        self._preserve_background = preserve_background

    def build(
        self,
        frame1: dai.Node.Output,
        frame2: dai.Node.Output,
        alpha: Optional[float] = None,
        preserve_background: Optional[bool] = None,
    ) -> "ImgFrameOverlay":
        """Configures the node connections.

        @param frame1: The input message for the background frame.
        @type frame1: dai.Node.Output
        @param frame2: The input message for the foreground frame.
        @type frame2: dai.Node.Output
        @param alpha: The weight of the background frame in the overlay.
        @type alpha: float
        @param preserve_background: If True, zero areas in the foreground frame are
            ignored in the output overlay frame.
        @type preserve_background: bool
        @return: The node object with the background and foreground streams overlaid.
        @rtype: ImgFrameOverlay
        """
        self.link_args(frame1, frame2)

        if alpha is not None:
            self.setAlpha(alpha)
        if preserve_background is not None:
            self.setPreserveBackground(preserve_background)

        self._logger.debug(
            f"ImgFrameOverlay built with alpha={alpha}, preserve_background={preserve_background}"
        )

        return self

    def process(self, frame1: dai.Buffer, frame2: dai.Buffer) -> None:
        """Processes incoming frames and overlays them.

        @param frame1: The input message for the background frame.
        @type frame1: dai.ImgFrame
        @param frame2: The input message for the foreground frame.
        @type frame2: dai.ImgFrame
        """
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
