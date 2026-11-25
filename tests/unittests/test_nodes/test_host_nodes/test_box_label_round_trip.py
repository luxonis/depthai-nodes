from collections.abc import Mapping

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import BoxClouds, BoxDetections
from depthai_nodes.node.box_associator.box_cloud_assembler import BoxCloudAssembler
from depthai_nodes.node.box_associator.box_labeler import BoxLabeler
from tests.utils import OutputMock
from tests.utils.box_scenes import (
    Cuboid,
    ViewPixels,
    box_detection,
    box_group,
    label_cloud,
    sample_faces,
)

pytest.importorskip("open3d")

FIRST = Cuboid((-300.0, 0.0, 100.0), (200.0, 150.0, 200.0))
SECOND = Cuboid((300.0, 0.0, 100.0), (200.0, 150.0, 200.0))


def label_view(
    view: int, boxes: Mapping[int, np.ndarray]
) -> tuple[ViewPixels, BoxDetections]:
    """Run the view's boxes through the labeler, one depth pixel per point."""
    points = np.concatenate(list(boxes.values()))
    mask = np.concatenate(
        [np.full(len(p), index, np.uint8) for index, p in enumerate(boxes.values())]
    )[None]
    depth = dai.ImgFrame()
    depth.setCvFrame(np.full(mask.shape, 1000, np.uint16), dai.ImgFrame.Type.RAW16)
    labeler = BoxLabeler().build(OutputMock(), OutputMock(), view=view)
    images = labeler.out.createOutputQueue()
    passthrough = labeler.passthroughDetections.createOutputQueue()

    labeler.process(BoxDetections([box_detection(t) for t in boxes], mask), depth)

    image, detections = images.get(), passthrough.get()
    assert isinstance(image, dai.ImgFrame)
    assert isinstance(detections, BoxDetections)
    return ViewPixels(image.getFrame(), points[None]), detections


def test_the_labelers_encoding_decodes_to_each_views_track_ids():
    left, left_detections = label_view(
        0,
        {
            70_000: sample_faces(FIRST, ["-x", "+z"], seed=0),
            3: sample_faces(SECOND, ["-x", "+z"], seed=1),
        },
    )
    right, right_detections = label_view(
        1,
        {
            5: sample_faces(FIRST, ["+x", "+z"], seed=2),
            9: sample_faces(SECOND, ["+x", "+z"], seed=3),
        },
    )
    assembler = BoxCloudAssembler().build(OutputMock())
    queue = assembler.out.createOutputQueue()

    assembler.process(
        box_group(label_cloud([right, left]), [left_detections, right_detections])
    )

    clouds = queue.get()
    assert isinstance(clouds, BoxClouds)
    assert {
        frozenset((d.view, d.track_id) for d in c.detections) for c in clouds.items
    } == {frozenset({(0, 70_000), (1, 5)}), frozenset({(0, 3), (1, 9)})}
