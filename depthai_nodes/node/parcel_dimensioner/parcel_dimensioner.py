"""Per-frame parcel fits against the detector's exact plane snapshot."""

import depthai as dai
import numpy as np

from depthai_nodes.algorithms import parcel as algorithm
from depthai_nodes.message.parcel import (
    ParcelCuboid,
    ParcelDetections,
    ParcelMeasurement,
    ParcelMeasurements,
)


class ParcelDimensioner(dai.node.HostNode):
    """Measure each eligible parcel using the single bundled fast baseline.

    ``out`` emits same-host ``ParcelMeasurements``. Each input parcel receives
    a no-fit, unknown-height, or measured outcome. Input frame failure and
    algorithm identity mismatch produce failed frames. The dimensioner uses
    only the plane in the batch, without independent plane state or tracking.
    """

    algorithm_id = algorithm.ALGORITHM_ID

    def __init__(self) -> None:
        super().__init__()
        self._components = None

    def build(self, parcels: dai.Node.Output) -> "ParcelDimensioner":
        """Connect a ``ParcelDetections`` output and bind the frozen measurement.

        Args:
            parcels: Same-host output of ``ParcelDetector``.

        Returns:
            This configured node.

        Raises:
            RuntimeError: If already built.
            ImportError: If the parcel extra dependencies are missing.
        """
        if self._components is not None:
            raise RuntimeError("ParcelDimensioner is already built")
        components = algorithm.create_components()
        self.link_args(parcels)
        self._components = components
        return self

    def process(self, parcels: dai.Buffer) -> None:
        """Measure a batch without extracting or copying its parcel point arrays."""
        if self._components is None:
            raise RuntimeError("Build ParcelDimensioner before processing")
        if not isinstance(parcels, ParcelDetections):
            raise TypeError("Expected ParcelDetections")
        common = dict(
            frame=parcels.frame,
            plane=parcels.plane,
            sources=parcels.sources,
            algorithm_id=parcels.algorithm_id,
        )
        failure = None
        results = []
        if parcels.algorithm_id != self.algorithm_id:
            failure = "algorithm_identity_mismatch"
        elif parcels.outcome == ParcelDetections.Outcome.FAILED:
            failure = parcels.failure_code
        else:
            try:
                plane = algorithm.SupportPlane(
                    parcels.plane.normal, parcels.plane.offset_mm, parcels.frame
                )
                for parcel in parcels.items:
                    cuboid = self._components.measure(
                        algorithm.PointCloud(parcel.xyz_mm, parcels.frame), plane
                    )
                    geometry = None
                    if cuboid is None:
                        status = ParcelMeasurement.Status.NO_FIT
                    else:
                        if not isinstance(cuboid, algorithm.Cuboid):
                            raise ValueError("Invalid cuboid result")
                        geometry = ParcelCuboid(
                            base_center_mm=cuboid.base_center_mm,
                            length_axis=cuboid.length_axis,
                            length_mm=cuboid.length_mm,
                            width_mm=cuboid.width_mm,
                            height_mm=cuboid.height_mm,
                        )
                        if (
                            abs(np.dot(geometry.length_axis, plane.normal)) > 1e-6
                            or abs(
                                np.dot(geometry.base_center_mm, plane.normal)
                                + plane.offset_mm
                            )
                            > 0.05
                        ):
                            raise ValueError("Cuboid must lie on the supplied plane")
                        status = (
                            ParcelMeasurement.Status.HEIGHT_UNKNOWN
                            if geometry.height_mm is None
                            else ParcelMeasurement.Status.MEASURED
                        )
                    results.append(
                        ParcelMeasurement(
                            candidate_id=parcel.candidate_id,
                            status=status,
                            cuboid=geometry,
                        )
                    )
            except Exception:
                failure = "measurement_failed"
        result = (
            ParcelMeasurements(
                **common,
                outcome=ParcelMeasurements.Outcome.FAILED,
                failure_code=failure,
            )
            if failure is not None
            else ParcelMeasurements(**common, items=tuple(results))
        )
        result.setTimestamp(parcels.getTimestamp())
        result.setTimestampDevice(parcels.getTimestampDevice())
        result.setSequenceNum(parcels.getSequenceNum())
        self.out.send(result)
