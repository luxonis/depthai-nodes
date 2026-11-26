"""Per-frame detection fits against the detector's exact plane snapshot."""

import depthai as dai
import numpy as np

from depthai_nodes.algorithms import measure_anything as algorithm
from depthai_nodes.message.measure_anything import (
    MeasureAnythingCuboid,
    MeasureAnythingDetections,
    MeasureAnythingMeasurement,
    MeasureAnythingMeasurements,
)


class MeasureAnythingDimensioner(dai.node.HostNode):
    """Fit dimensions for each eligible object emitted by the detector.

    Install ``depthai-nodes[measure-anything]`` and connect a
    ``MeasureAnythingDetections`` output through ``build()``. This host node
    consumes each object's selected, read-only point array without another
    extraction or copy. Both nodes bind to the same frozen algorithm and
    configuration. A different algorithm identity produces a failed frame.

    Measurement uses the exact upward ``SupportPlane`` and calibrated frame
    stored in the input batch. It has no independent plane/config input, so a
    newer detector configuration cannot change measurement of a queued batch.
    The bundled algorithm fits a cuboid footprint and height in millimeters;
    its published accuracy applies to the evaluated object and input profile.
    The packaged ``visualization.html`` in
    ``depthai_nodes/algorithms/measure_anything/implementation/`` explains the
    geometry, real examples and measurement limits.

    ``out`` emits ``MeasureAnythingMeasurements`` with one
    ``MeasureAnythingMeasurement`` per eligible input object, preserving order
    and frame-local candidate IDs. Its ``Status`` is ``NO_FIT`` when there is no
    cuboid, ``HEIGHT_UNKNOWN`` for a footprint without height, or ``MEASURED``
    for a complete fit. A ``MeasureAnythingCuboid`` contains base center, length
    axis, length, width and optional height. Results have no tracking or temporal
    aggregation.

    Input frame failures propagate as failed measurement frames. Computation or
    invalid-geometry errors fail the whole frame; an empty successful input
    produces an empty successful output. Frame timestamps, sequence, source
    timing and algorithm identity are retained. These messages contain Python
    values for one host runtime; XLink and RemoteConnection do not serialize
    their nested fields.

    Example:
        With an existing ``MeasureAnythingDetector`` node::

            from depthai_nodes.node import MeasureAnythingDimensioner

            dimensioner = pipeline.create(MeasureAnythingDimensioner).build(detector.out)
            results = dimensioner.out.createOutputQueue()
    """

    algorithm_id = algorithm.ALGORITHM_ID

    def __init__(self) -> None:
        super().__init__()
        self._components = None

    def build(self, detections: dai.Node.Output) -> "MeasureAnythingDimensioner":
        """Connect a ``MeasureAnythingDetections`` output and bind the frozen
        measurement.

        Args:
            detections: Same-host output of ``MeasureAnythingDetector``.

        Returns:
            This configured node.

        Raises:
            RuntimeError: If already built.
            ImportError: If the measure-anything extra dependencies are missing.
        """
        if self._components is not None:
            raise RuntimeError("MeasureAnythingDimensioner is already built")
        components = algorithm.create_components()
        self.link_args(detections)
        self._components = components
        return self

    def process(self, detections: dai.Buffer) -> None:
        """Measure a batch without extracting or copying its detection point arrays."""
        if self._components is None:
            raise RuntimeError("Build MeasureAnythingDimensioner before processing")
        if not isinstance(detections, MeasureAnythingDetections):
            raise TypeError("Expected MeasureAnythingDetections")
        common = dict(
            frame=detections.frame,
            plane=detections.plane,
            sources=detections.sources,
            algorithm_id=detections.algorithm_id,
        )
        failure = None
        results = []
        if detections.algorithm_id != self.algorithm_id:
            failure = "algorithm_identity_mismatch"
        elif detections.outcome == MeasureAnythingDetections.Outcome.FAILED:
            failure = detections.failure_code
        else:
            try:
                supportPlane = algorithm.SupportPlane(
                    detections.plane.normal,
                    detections.plane.offset_mm,
                    detections.frame,
                )
                for detection in detections.items:
                    cuboid = self._components.measure(
                        algorithm.PointCloud(detection.xyz_mm, detections.frame),
                        supportPlane,
                    )
                    geometry = None
                    if cuboid is None:
                        status = MeasureAnythingMeasurement.Status.NO_FIT
                    else:
                        if not isinstance(cuboid, algorithm.Cuboid):
                            raise ValueError("Invalid cuboid result")
                        geometry = MeasureAnythingCuboid(
                            base_center_mm=cuboid.base_center_mm,
                            length_axis=cuboid.length_axis,
                            length_mm=cuboid.length_mm,
                            width_mm=cuboid.width_mm,
                            height_mm=cuboid.height_mm,
                        )
                        if (
                            abs(np.dot(geometry.length_axis, supportPlane.normal))
                            > 1e-6
                            or abs(
                                np.dot(geometry.base_center_mm, supportPlane.normal)
                                + supportPlane.offset_mm
                            )
                            > 0.05
                        ):
                            raise ValueError("Cuboid must lie on the supplied plane")
                        status = (
                            MeasureAnythingMeasurement.Status.HEIGHT_UNKNOWN
                            if geometry.height_mm is None
                            else MeasureAnythingMeasurement.Status.MEASURED
                        )
                    results.append(
                        MeasureAnythingMeasurement(
                            candidate_id=detection.candidate_id,
                            status=status,
                            cuboid=geometry,
                        )
                    )
            except Exception:
                failure = "measurement_failed"
        result = (
            MeasureAnythingMeasurements(
                **common,
                outcome=MeasureAnythingMeasurements.Outcome.FAILED,
                failure_code=failure,
            )
            if failure is not None
            else MeasureAnythingMeasurements(**common, items=tuple(results))
        )
        result.setTimestamp(detections.getTimestamp())
        result.setTimestampDevice(detections.getTimestampDevice())
        result.setSequenceNum(detections.getSequenceNum())
        self.out.send(result)
