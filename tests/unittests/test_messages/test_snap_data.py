import depthai as dai
import pytest

from depthai_nodes.message import SnapData


@pytest.fixture
def img_frame() -> dai.ImgFrame:
    frame = dai.ImgFrame()
    frame.setSequenceNum(1)
    return frame


@pytest.fixture
def img_detections() -> dai.ImgDetections:
    dets = dai.ImgDetections()
    det = dai.ImgDetection()
    det.label = 1
    det.confidence = 0.9
    dets.detections = [det]
    return dets


def test_snap_data_initialization_minimal(img_frame: dai.ImgFrame):
    snap = SnapData("test_snap", img_frame)
    assert isinstance(snap, SnapData)
    assert snap.snap_name == "test_snap"
    assert snap.file_name == ""
    assert snap.frame is img_frame
    assert snap.detections is None
    assert snap.tags == []
    assert snap.extras == {}


def test_snap_data_initialization_full(
    img_frame: dai.ImgFrame, img_detections: dai.ImgDetections
):
    tags = ["triggered", "low_conf"]
    extras = {"reason": "low_confidence", "model": "yolo"}

    snap = SnapData(
        snap_name="snap1",
        frame=img_frame,
        file_name="snap1_001.jpg",
        detections=img_detections,
        tags=tags,
        extras=extras,
    )

    assert snap.snap_name == "snap1"
    assert snap.file_name == "snap1_001.jpg"
    assert snap.frame is img_frame
    assert snap.detections is img_detections
    assert snap.tags == tags
    assert snap.extras == extras


def test_snap_data_tags_and_extras_mutability(img_frame: dai.ImgFrame):
    s1 = SnapData("a", img_frame)
    s2 = SnapData("b", img_frame)
    s1.tags.append("one")
    s1.extras["x"] = "1"
    assert s2.tags == []
    assert s2.extras == {}


def test_snap_data_detections_type(img_frame: dai.ImgFrame):
    dets = dai.ImgDetections()
    snap = SnapData("snap", img_frame, detections=dets)
    assert isinstance(snap.detections, dai.ImgDetections)

    snap_no_det = SnapData("snap2", img_frame, detections=None)
    assert snap_no_det.detections is None


def test_snap_data_can_store_and_retrieve_all_fields(img_frame: dai.ImgFrame):
    snap = SnapData("snap_final", img_frame)
    snap.file_name = "final.jpg"
    snap.tags = ["t1", "t2"]
    snap.extras = {"key": "value"}

    assert snap.file_name == "final.jpg"
    assert snap.tags == ["t1", "t2"]
    assert snap.extras == {"key": "value"}
