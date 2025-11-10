import os
from unittest.mock import MagicMock

import depthai as dai
import pytest

from depthai_nodes.message import SnapData
from depthai_nodes.node import SnapsUploader
from tests.utils.nodes.mocks import OutputMock


@pytest.fixture
def frame() -> dai.ImgFrame:
    frame = dai.ImgFrame()
    frame.setSequenceNum(1)
    return frame


@pytest.fixture
def detections() -> dai.ImgDetections:
    dets = dai.ImgDetections()
    det = dai.ImgDetection()
    det.label = 1
    det.confidence = 0.9
    dets.detections = [det]
    return dets


@pytest.fixture
def snap_data(frame, detections) -> SnapData:
    return SnapData(
        snap_name="test_snap",
        file_name="test_snap_001.jpg",
        frame=frame,
        detections=detections,
        tags=["low_conf"],
        extras={"reason": "unit_test"},
    )


@pytest.fixture
def uploader() -> SnapsUploader:
    uploader = SnapsUploader()
    uploader._logger = MagicMock()
    uploader._em = MagicMock()
    return uploader


def test_build(uploader):
    mock_output = OutputMock()
    uploader.build(mock_output)


def test_set_token_and_url(monkeypatch):
    uploader = SnapsUploader()

    uploader.set_token("test_token")
    assert os.environ["DEPTHAI_HUB_API_KEY"] == "test_token"

    uploader.set_url("test-url")
    assert os.environ["DEPTHAI_HUB_EVENTS_BASE_URL"] == "test-url"

    # Repeated calls shouldn't overwrite (uses setdefault)
    uploader.set_token("new_token")
    assert os.environ["DEPTHAI_HUB_API_KEY"] == "test_token"


def test_process_success(uploader, snap_data):
    """Test successful snap upload."""
    uploader._em.sendSnap.return_value = True

    uploader.process(snap_data)

    uploader._em.sendSnap.assert_called_once()
    uploader._logger.info.assert_called_with("Snap 'test_snap' sent successfully.")


def test_process_failure(uploader, snap_data):
    """Test failed snap upload logging."""
    uploader._em.sendSnap.return_value = False

    uploader.process(snap_data)

    uploader._logger.error.assert_called_with("Failed to send snap 'test_snap'.")


def test_process_invalid_type(uploader):
    """Should raise if message is not SnapData."""
    with pytest.raises(AssertionError):
        uploader.process(dai.Buffer())


def test_environment_variable_setting(monkeypatch):
    """Test that set_token and set_url correctly set environment variables."""
    monkeypatch.delenv("DEPTHAI_HUB_API_KEY", raising=False)
    monkeypatch.delenv("DEPTHAI_HUB_EVENTS_BASE_URL", raising=False)

    uploader = SnapsUploader()
    uploader.set_token("my_token")
    uploader.set_url("my_url")

    assert os.environ["DEPTHAI_HUB_API_KEY"] == "my_token"
    assert os.environ["DEPTHAI_HUB_EVENTS_BASE_URL"] == "my_url"
