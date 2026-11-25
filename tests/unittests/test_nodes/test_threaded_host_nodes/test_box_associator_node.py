from typing import Any, cast

import depthai as dai
import numpy as np
import pytest

pytest.importorskip("open3d")

from depthai_nodes.message import BoxDetections  # noqa: E402
from depthai_nodes.node import BoxAssociator  # noqa: E402
from depthai_nodes.node.box_associator.box_associator import MAX_VIEWS  # noqa: E402
from depthai_nodes.node.box_associator.box_cloud_assembler import (  # noqa: E402
    CLOUD_KEY,
    BoxCloudAssembler,
    detections_key,
)
from depthai_nodes.node.box_associator.box_labeler import BoxLabeler  # noqa: E402
from tests.utils import OutputMock, PipelineMock, PointCloudMock, SyncMock  # noqa: E402
from tests.utils.box_scenes import box_detection  # noqa: E402


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> PipelineMock:
    monkeypatch.setattr(dai.node, "PointCloud", PointCloudMock)
    return PipelineMock()


def create(pipeline: PipelineMock) -> BoxAssociator:
    return cast(BoxAssociator, pipeline.create(cast(Any, BoxAssociator)))


def views(count: int) -> list[tuple[OutputMock, OutputMock]]:
    return [(OutputMock(), OutputMock()) for _ in range(count)]


def build(pipeline: PipelineMock, view_outputs) -> BoxAssociator:
    return create(pipeline).build(cast(Any, view_outputs))


def nodes_of(pipeline: PipelineMock, node_type: type) -> list:
    return [n for n in pipeline.getAllNodes() if isinstance(n, node_type)]


def find_node(pipeline: PipelineMock, node_type: type):
    (node,) = nodes_of(pipeline, node_type)
    return node


def pairing_sync(pipeline: PipelineMock) -> SyncMock:
    return find_node(pipeline, SyncMock)


def labeler_of(pipeline: PipelineMock, detections, depth) -> BoxLabeler:
    (labeler,) = [
        n
        for n in nodes_of(pipeline, BoxLabeler)
        if n._linked_args == (detections, depth)
    ]
    return labeler


def painted_view(labeler: BoxLabeler) -> int:
    """Return the view a labeler paints, read back from its label image's blue."""
    queue = labeler.out.createOutputQueue()
    depth = dai.ImgFrame()
    depth.setCvFrame(np.zeros((1, 1), np.uint16), dai.ImgFrame.Type.RAW16)
    labeler.process(
        BoxDetections([box_detection(1)], np.zeros((1, 1), np.uint8)), depth
    )
    return int(queue.get().getFrame()[0, 0, 2]) - 1


def test_each_view_is_wired_by_its_index(pipeline: PipelineMock):
    view_outputs = views(3)

    build(pipeline, view_outputs)

    cloud = find_node(pipeline, PointCloudMock)
    sync = pairing_sync(pipeline)
    for i, (detections, depth) in enumerate(view_outputs):
        labeler = labeler_of(pipeline, detections, depth)
        assert painted_view(labeler) == i
        assert cloud.getDepthInput(str(i)) in depth.links
        assert cloud.getColorInput(str(i)) in labeler.out.links
        assert sync.inputs[detections_key(i)] in labeler.passthroughDetections.links
    assert sync.inputs[CLOUD_KEY] in cloud.outputPointCloud.links


def test_the_label_cloud_is_organized_and_both_syncs_run_on_the_host_untuned(
    pipeline: PipelineMock,
):
    associator = build(pipeline, views(2))

    cloud = find_node(pipeline, PointCloudMock)
    assert cloud.device is None
    assert cloud.initialConfig.getOrganized()
    for sync in (cloud.sync, pairing_sync(pipeline)):
        assert sync.runOnHost()
        assert not sync.syncThresholdSet
    assembler = find_node(pipeline, BoxCloudAssembler)
    assert assembler._linked_args == (pairing_sync(pipeline).out,)
    # The pipeline mock shadows ``out``, so read the node's own property.
    assert BoxAssociator.out.fget(associator) is assembler.out


@pytest.mark.parametrize("count", [0, MAX_VIEWS + 1])
def test_build_rejects_a_view_count_out_of_range_creating_nothing(
    pipeline: PipelineMock, count: int
):
    associator = create(pipeline)

    with pytest.raises(ValueError, match="views"):
        associator.build(cast(Any, views(count)))

    assert pipeline.getAllNodes() == [associator]


def test_build_accepts_the_most_views(pipeline: PipelineMock):
    build(pipeline, views(MAX_VIEWS))

    assert len(nodes_of(pipeline, BoxLabeler)) == MAX_VIEWS


def test_a_second_build_raises(pipeline: PipelineMock):
    associator = build(pipeline, views(1))
    nodes = pipeline.getAllNodes()

    with pytest.raises(RuntimeError, match="already built"):
        associator.build(cast(Any, views(1)))

    assert pipeline.getAllNodes() == nodes
