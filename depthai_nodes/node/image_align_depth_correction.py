"""Shared RGB target-axial correction between ImageAlign and PointCloud."""

import depthai as dai

from depthai_nodes.utils.axial_depth import AxialDepthCorrector, AxialDepthPolicy


class ImageAlignDepthCorrection(dai.node.ThreadedHostNode):
    """Convert aligned source-camera axial Z into target-camera axial depth.

    Connect ``outputAligned`` and the exact ``passthroughInput`` from one
    ImageAlign node. Both bounded blocking inputs must remain lossless. Sequence
    number, host timestamp and device timestamp must match exactly. A mismatched
    pair fails visibly. Missing inputs block until messages arrive, so keep both
    streams lossless. Preserve unique occurrence sequence numbers when replaying.
    ImageAlign owns the target-image match, so no third image input is needed.

    The aligned frame's ``ImgTransformation`` describes the actual target grid.
    It must be an undistorted perspective grid. The source passthrough must
    retain the depth optical frame's actual calibration and rectification;
    a native left image cannot substitute for rectified depth metadata.
    Branch RGB undistortion before alignment when needed, retaining separate
    native image streams for detection. Source and target calibration references,
    sizes, intrinsics, rigid poses and RAW16 layouts are checked per frame.

    Select ``policy`` explicitly in ``build()``. Use
    ``AxialDepthPolicy.SOURCE_AXIAL_REQUIRES_CORRECTION`` for the source-Z
    ImageAlign behavior verified with DepthAI 3.12.0. Use
    ``AxialDepthPolicy.TARGET_AXIAL_PASSTHROUGH`` only for a producer verified
    to already provide target-camera Z. There is no SDK-version heuristic.
    Recheck the producer's behavior when upgrading the SDK.

    ``out`` emits an independent native RAW16 ``ImgFrame`` in target-camera
    axial millimeters for PointCloud. ``result`` emits an ``AxialDepthFrameResult``
    containing that frame and its provenance. Use ``result.provenance.to_dict()``
    for reports. Geometry identity, axial conventions, correction method,
    invalid-value/quantization rules and source association are retained.
    The structured result stays in one Python runtime, including an OAK-4 app's
    CPU. XLink, RemoteConnection and native frame cloning do not retain its
    Python fields. Keep the matching result or sidecar at later correction
    boundaries. Structured results reject double conversion; an unmarked bare
    native frame cannot reveal its earlier conversion history.

    Conversion uses float64 arithmetic and nearest-integer millimeter rounding
    with ties to even. Depth 0 and 65535 remain invalid; nonfinite, nonpositive
    or out-of-range conversions become zero. Input buffers remain unchanged.
    The cloned output preserves target transformation, source metadata, timing,
    sequence, active-plane offset, row stride and padding. PointCloud consumers
    still need to exclude both invalid depths and nonfinite points, preserve
    raster order when filtering, and select the intended millimeter coordinate
    frame explicitly.

    This Python node runs on the workstation in a host pipeline or on the OAK-4
    application CPU in a standalone app. It does not require an implicit device.
    The repository's ``examples/image_align_depth_correction`` fixture verifies
    alignment and corrected cloud geometry on the host or OAK-4. Its synthetic
    result does not establish live NeuralDepth accuracy or throughput.
    For array-based correction and explicit previous provenance, use
    ``depthai_nodes.utils.axial_depth.AxialDepthCorrector``.

    Example:
        With configured ImageAlign and PointCloud nodes::

            from depthai_nodes.node import ImageAlignDepthCorrection
            from depthai_nodes.utils.axial_depth import AxialDepthPolicy

            correction = pipeline.create(ImageAlignDepthCorrection).build(
                align.outputAligned,
                align.passthroughInput,
                policy=AxialDepthPolicy.SOURCE_AXIAL_REQUIRES_CORRECTION,
            )
            correction.out.link(cloud.inputDepth)
            provenance = correction.result.createOutputQueue()
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
