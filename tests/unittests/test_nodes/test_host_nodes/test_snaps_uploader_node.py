import os
from unittest.mock import MagicMock

import depthai as dai
import numpy as np
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
    frame.setData(np.zeros((10, 10, 3), dtype=np.uint8))
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
    fg = dai.FileGroup()
    fg.addImageDetectionsPair("test_snap_001", frame, detections)
    return SnapData(
        snap_name="test_snap",
        file_group=fg,
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


def test_environment_variable_setting(monkeypatch):
    monkeypatch.delenv("DEPTHAI_HUB_API_KEY", raising=False)

    uploader = SnapsUploader()
    uploader.setToken("my_token")

    assert os.environ["DEPTHAI_HUB_API_KEY"] == "my_token"


def test_set_cache_dir(uploader, mock_snaps_uploader_logger):
    uploader.setCacheDir("/custom/cache/path")

    uploader._em.setCacheDir.assert_called_once_with("/custom/cache/path")
    mock_snaps_uploader_logger.info.assert_called_with(
        "Set cache directory to: /custom/cache/path"
    )


def test_set_cache_if_cannot_send(uploader, mock_snaps_uploader_logger):
    uploader.setCacheIfCannotSend(True)

    uploader._em.setCacheIfCannotSend.assert_called_once_with(True)
    mock_snaps_uploader_logger.info.assert_called_with(
        "Cache snaps if they cannot be uploaded: True"
    )


def test_set_log_response(uploader, mock_snaps_uploader_logger):
    uploader.setLogResponse(True)

    uploader._em.setLogResponse.assert_called_once_with(True)
    mock_snaps_uploader_logger.info.assert_called_with("Log server responses: True")


def test_process_success(uploader, snap_data, mock_snaps_uploader_logger):
    uploader._em.sendSnap.return_value = True
    uploader.process(snap_data)

    uploader._em.sendSnap.assert_called_once_with(
        name="test_snap",
        fileGroup=snap_data.file_group,
        tags=["low_conf"],
        extras={"reason": "unit_test"},
    )
    mock_snaps_uploader_logger.info.assert_called_with(
        "Snap 'test_snap' sent successfully."
    )


def test_process_failure(uploader, snap_data, mock_snaps_uploader_logger):
    uploader._em.sendSnap.return_value = False
    uploader.process(snap_data)

    mock_snaps_uploader_logger.error.assert_called_with(
        "Failed to send snap 'test_snap'."
    )


def test_process_invalid_type(uploader):
    with pytest.raises(AssertionError):
        uploader.process(dai.Buffer())
