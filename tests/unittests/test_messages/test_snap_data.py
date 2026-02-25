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
    fg = dai.FileGroup()
    fg.addFile("image", img_frame)

    snap = SnapData("test_snap", fg)

    assert isinstance(snap, SnapData)
    assert snap.snap_name == "test_snap"
    assert snap.file_group is fg
    assert snap.tags == []
    assert snap.extras == {}


def test_snap_data_initialization_full(
    img_frame: dai.ImgFrame, img_detections: dai.ImgDetections
):
    fg = dai.FileGroup()
    fg.addImageDetectionsPair("capture", img_frame, img_detections)
    tags = ["triggered", "low_conf"]
    extras = {"reason": "low_confidence", "model": "yolo"}

    snap = SnapData("snap1", fg, tags=tags, extras=extras)

    assert snap.snap_name == "snap1"
    assert snap.file_group is fg
    assert snap.tags == tags
    assert snap.extras == extras


def test_snap_data_empty_file_group():
    fg = dai.FileGroup()
    snap = SnapData("empty_snap", fg)

    assert snap.snap_name == "empty_snap"
    assert snap.file_group is fg
    assert snap.tags == []
    assert snap.extras == {}


def test_snap_data_multiple_files_in_group(
    img_frame: dai.ImgFrame, img_detections: dai.ImgDetections
):
    fg = dai.FileGroup()
    fg.addFile("image1", img_frame)
    fg.addFile("detections", img_detections)

    snap = SnapData(
        snap_name="multi_file_snap",
        file_group=fg,
        tags=["multi"],
        extras={"count": "2"},
    )

    assert snap.snap_name == "multi_file_snap"
    assert snap.file_group is fg
    assert snap.tags == ["multi"]
    assert snap.extras == {"count": "2"}
