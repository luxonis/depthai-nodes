"""Run both parcel nodes on a synthetic calibrated scene without an OAK device.

Install ``depthai-nodes[parcel]`` and run this file with Python. The scene has a
220 by 160 mm parcel with a 110 mm height. No external checkout, model download,
source recording, or Workbench connection is used.
"""

from datetime import timedelta

import depthai as dai
import numpy as np

from depthai_nodes.message import (
    ParcelMeasurement,
    ParcelMeasurements,
    ParcelPlane,
    SupportPlane,
)
from depthai_nodes.node import ParcelDetector, ParcelDimensioner


def make_scene():
    """Return a cloud, three images with native calibration, and a supplied plane."""
    size, focal, center = 260, 500.0, 129.5
    v, u = np.indices((size, size))
    rays = np.stack(
        ((u - center) / focal, (v - center) / focal, np.ones_like(u)), axis=-1
    )
    images = []
    cloud = dai.PointCloudData()
    for role, camera_x in (("color", 0), ("left", -15), ("right", 15)):
        top = rays[:, :, :2] * 890 + np.array([camera_x, 0])
        parcel = (np.abs(top[:, :, 0]) <= 110) & (np.abs(top[:, :, 1]) <= 80)
        pixels = np.where(parcel, 185, 35).astype(np.uint8)
        image = dai.ImgFrame()
        if role == "color":
            image.setType(dai.ImgFrame.Type.BGR888i)
            image.setFrame(np.repeat(pixels[:, :, None], 3, axis=2))
            xyz = rays * np.where(parcel, 890.0, 1000.0)[:, :, None]
            cloud.setPoints(xyz.reshape(-1, 3).astype(np.float32))
        else:
            image.setType(dai.ImgFrame.Type.GRAY8)
            image.setFrame(pixels)
        image.setWidth(size)
        image.setHeight(size)
        images.append(image)
        extrinsics = dai.Extrinsics(
            np.eye(3).tolist(),
            dai.Point3f(camera_x, 0, 0),
            dai.CameraBoardSocket.CAM_A,
            dai.LengthUnit.MILLIMETER,
        )
        extrinsics.toDeviceId = "synthetic"
        transformation = dai.ImgTransformation(
            size,
            size,
            ((focal, 0, center), (0, focal, center), (0, 0, 1)),
            dai.CameraModel.Perspective,
            [],
            extrinsics,
        )
        image.setTransformation(transformation)
        if role == "color":
            cloud.setTransformation(transformation)
    plane = ParcelPlane(
        frame="scene",
        plane=SupportPlane(normal=(0, 0, -1), offset_mm=1000),
        down=(0, 0, 1),
    )
    messages = (cloud, *images, plane)
    for message in messages:
        message.setSequenceNum(1)
        message.setTimestamp(timedelta(seconds=1))
        message.setTimestampDevice(timedelta(seconds=1))
    return messages


class SyntheticScene(dai.node.ThreadedHostNode):
    """Emit a complete scene once into the native host pipeline."""

    def __init__(self):
        super().__init__()
        self.streams = tuple(self.createOutput(f"source_{i}") for i in range(5))
        self.messages = make_scene()

    def run(self):
        for output, message in zip(self.streams, self.messages):
            output.send(message)


def main():
    """Run and assert one measured parcel from the native same-host pipeline."""
    with dai.Pipeline(False) as pipeline:
        source = pipeline.create(SyntheticScene)
        detector = pipeline.create(ParcelDetector).build(*source.streams)
        dimensioner = pipeline.create(ParcelDimensioner).build(detector.out)
        results = dimensioner.out.createOutputQueue()
        pipeline.start()
        result = results.get(timedelta(seconds=15))
        if result is None:
            raise RuntimeError("Synthetic parcel pipeline did not return a frame")
        assert result.outcome == ParcelMeasurements.Outcome.SUCCESS, result.failure_code
        assert result.getSequenceNum() == 1
        assert len(result.items) == 1
        item = result.items[0]
        assert item.status == ParcelMeasurement.Status.MEASURED
        geometry = item.cuboid
        np.testing.assert_allclose(
            [geometry.length_mm, geometry.width_mm, geometry.height_mm],
            [220, 160, 110],
            atol=5,
            rtol=0,
        )
        print(f"candidate={item.candidate_id} algorithm={result.algorithm_id}")
        print(
            f"length={geometry.length_mm:.3f} width={geometry.width_mm:.3f} height={geometry.height_mm:.3f} mm"
        )
        pipeline.stop()


if __name__ == "__main__":
    main()
