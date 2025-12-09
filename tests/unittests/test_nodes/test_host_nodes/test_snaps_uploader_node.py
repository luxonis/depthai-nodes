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
    frame.setWidth(10)
    frame.setHeight(10)
    frame.setSequenceNum(1)

    # Create dummy image data
    import numpy as np

    img = np.zeros((10, 10, 3), dtype=np.uint8)

    frame.setData(img)
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


@pytest.fixture()
def mock_snaps_uploader_logger(monkeypatch):
    """Automatically mock the module-level logger used inside SnapsUploader for all
    tests in this file."""
    mock_logger = MagicMock()
    monkeypatch.setattr("depthai_nodes.node.snaps_uploader.logger", mock_logger)
    return mock_logger


def test_build(uploader):
    mock_output = OutputMock()
    uploader.build(mock_output)


def test_set_token_and_url(monkeypatch):
    uploader = SnapsUploader()

    uploader.set_token("test_token")
    assert os.environ["DEPTHAI_HUB_API_KEY"] == "test_token"

    uploader.set_url("test-url")
    assert os.environ["DEPTHAI_HUB_EVENTS_BASE_URL"] == "test-url"

    uploader.set_token("new_token")
    assert os.environ["DEPTHAI_HUB_API_KEY"] == "test_token"


def test_process_success(uploader, snap_data, mock_snaps_uploader_logger):
    """Test successful snap upload."""
    uploader._em.sendSnap.return_value = True
    uploader.process(snap_data)

    uploader._em.sendSnap.assert_called_once()
    mock_snaps_uploader_logger.info.assert_called_with(
        "Snap 'test_snap' sent successfully."
    )


def test_process_failure(uploader, snap_data, mock_snaps_uploader_logger):
    """Test failed snap upload logging."""
    uploader._em.sendSnap.return_value = False
    uploader.process(snap_data)

    mock_snaps_uploader_logger.error.assert_called_with(
        "Failed to send snap 'test_snap'."
    )


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
