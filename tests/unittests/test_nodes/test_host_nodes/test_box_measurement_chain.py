from typing import cast

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import (
    BoxClouds,
    BoxDetections,
    BoxMeasurements,
    MeasurementStatus,
    SupportPlane,
)
from depthai_nodes.node import BoxDimensioner
from depthai_nodes.node.box_associator.box_cloud_assembler import BoxCloudAssembler
from depthai_nodes.node.box_associator.box_labeler import BoxLabeler
from tests.utils import OutputMock
from tests.utils.box_scenes import ViewPixels, box_detection, box_group, label_cloud

pytest.importorskip("open3d")

CLOUD_SIZE = (64, 48)
FLOOR = SupportPlane(normal=(0, 0, 1), offset_mm=-1000)
TRACK_ID = 5


def output() -> dai.Node.Output:
    return cast(dai.Node.Output, OutputMock())


def flat_top_scene() -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel points of a floor at 1000 mm with a top 100 mm above it, and a mask
    whose track (index 0) covers that top."""
    width, rows = CLOUD_SIZE
    x, y = np.meshgrid(np.linspace(-200, 200, width), np.linspace(-150, 150, rows))
    z = np.full((rows, width), 1000.0)
    z[10:38, 14:50] = 900.0
    mask = np.full((24, 32), 255, np.uint8)
    mask[5:19, 7:25] = 0
    return np.dstack((x, y, z)).astype(np.float32), mask


def test_one_view_is_measured_through_labeler_associator_and_dimensioner():
    points, mask = flat_top_scene()
    width, rows = CLOUD_SIZE
    labeler = BoxLabeler().build(output(), output(), view=0)
    labels = labeler.out.createOutputQueue()
    passthrough = labeler.passthroughDetections.createOutputQueue()
    assembler = BoxCloudAssembler().build(output())
    box_clouds = assembler.out.createOutputQueue()
    dimensioner = BoxDimensioner().build(output(), FLOOR)
    measurements = dimensioner.out.createOutputQueue()

    results = []
    for _ in range(2):
        depth = dai.ImgFrame()
        depth.setCvFrame(
            np.full((rows, width), 1000, np.uint16), dai.ImgFrame.Type.RAW16
        )
        labeler.process(BoxDetections([box_detection(TRACK_ID)], mask), depth)
        image, detections = labels.get(), passthrough.get()
        assert isinstance(image, dai.ImgFrame)
        assert isinstance(detections, BoxDetections)
        cloud = label_cloud([ViewPixels(image.getFrame(), points)])
        assembler.process(box_group(cloud, [detections]))
        clouds = box_clouds.get()
        assert isinstance(clouds, BoxClouds)
        dimensioner.process(clouds)
        message = measurements.get()
        assert isinstance(message, BoxMeasurements)
        (result,) = message.items
        results.append(result)

    for result in results:
        assert result.measurement_status == MeasurementStatus.MEASURED
        assert result.dimensions is not None
        assert result.dimensions.height_mm == pytest.approx(100, abs=0.05)
        ((view, track_id),) = [(d.view, d.track_id) for d in result.detections]
        assert (view, track_id) == (0, TRACK_ID)
    assert results[0].box_id == results[1].box_id
