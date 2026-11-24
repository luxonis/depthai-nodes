"""Shared RGB target-axial correction between ImageAlign and PointCloud."""

import depthai as dai

from depthai_nodes.utils.axial_depth import AxialDepthCorrector, AxialDepthPolicy


class ImageAlignDepthCorrection(dai.node.ThreadedHostNode):
    """Convert aligned source axial Z before native PointCloud generation.

    This Python node runs on the workstation for a host pipeline, or on the OAK-4
    application CPU inside a standalone app. It needs no implicit camera device.
    Both bounded blocking inputs must originate from one ImageAlign node. The
    source input is its exact ``passthroughInput``, including rectified geometry.
    Wrong sequence or timestamps fail visibly rather than selecting nearby frames.

    ``out`` emits native RAW16 target-camera axial millimeters. ``result`` emits
    same-runtime ``AxialDepthFrameResult`` containing that frame and its provenance.
    Retain the associated result for reporting or explicit double-conversion checks.
    Native frame copies and XLink do not preserve the Python provenance.

    Example:
        With configured ImageAlign and PointCloud nodes::

            correction = pipeline.create(ImageAlignDepthCorrection).build(
                align.outputAligned,
                align.passthroughInput,
                policy=AxialDepthPolicy.SOURCE_AXIAL_REQUIRES_CORRECTION,
            )
            correction.out.link(cloud.inputDepth)
    """

    def __init__(self) -> None:
        super().__init__()
        self._input = self.createInput("alignedDepth", blocking=True, queueSize=4)
        self._source = self.createInput("sourceDepth", blocking=True, queueSize=4)
        self._out = self.createOutput("out")
        self._result = self.createOutput("result")
        self._corrector: AxialDepthCorrector | None = None

    @property
    def out(self) -> dai.Node.Output:
        """Native target-axial RAW16 millimeters for PointCloud."""
        return self._out

    @property
    def result(self) -> dai.Node.Output:
        """Associated frame and provenance for consumers in the same Python runtime."""
        return self._result

    def build(
        self,
        alignedDepth: dai.Node.Output,
        sourceDepth: dai.Node.Output,
        *,
        policy: AxialDepthPolicy | str,
    ) -> "ImageAlignDepthCorrection":
        """Connect the aligned output and exact source passthrough of one ImageAlign.

        Args:
            alignedDepth: ImageAlign ``outputAligned`` on an undistorted target grid.
            sourceDepth: The same ImageAlign node's ``passthroughInput``.
            policy: Explicit correction or target-axial passthrough declaration.

        Returns:
            This configured node.

        Raises:
            ValueError: If policy or stream ownership is invalid.
            RuntimeError: If already built.
        """
        if self._corrector is not None:
            raise RuntimeError("ImageAlignDepthCorrection is already built")
        corrector = AxialDepthCorrector(policy=policy)
        producer = alignedDepth.getParent()
        if (
            not isinstance(producer, dai.node.ImageAlign)
            or sourceDepth.getParent() is not producer
            or alignedDepth.getName() != producer.outputAligned.getName()
            or sourceDepth.getName() != producer.passthroughInput.getName()
        ):
            raise ValueError(
                "Use outputAligned and passthroughInput from one ImageAlign"
            )
        alignedDepth.link(self._input)
        sourceDepth.link(self._source)
        self._corrector = corrector
        return self

    def run(self) -> None:
        """Process lossless source-associated pairs with bounded queue backpressure."""
        if self._corrector is None:
            raise RuntimeError("Build ImageAlignDepthCorrection before running")
        while self.isRunning():
            aligned = self._input.get()
            source = self._source.get()
            result = self._corrector.correct_frame(aligned, source)
            self._out.send(result.frame)
            self._result.send(result)
