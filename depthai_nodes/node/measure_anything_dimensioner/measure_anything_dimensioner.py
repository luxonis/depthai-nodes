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
    """Measure each eligible detection using the single bundled fast baseline.

    ``out`` emits same-host ``MeasureAnythingMeasurements``. Each input detection receives
    a no-fit, unknown-height, or measured outcome. Input frame failure and
    algorithm identity mismatch produce failed frames. The dimensioner uses
    only the plane in the batch, without independent plane state or tracking.
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
