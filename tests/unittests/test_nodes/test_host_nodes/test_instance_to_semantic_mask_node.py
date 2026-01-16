import time

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import ImgDetectionsExtended
from depthai_nodes.node import InstanceToSemanticMask
from tests.utils import (
    LOG_INTERVAL,
    OutputMock,
    create_img_detections_extended,
)


@pytest.fixture(scope="session")
def duration(request):
    d = request.config.getoption("--duration")
    if d is None:
        return 1e-6
    return d


@pytest.fixture
def converter():
    return InstanceToSemanticMask()


def _instance_to_semantic_mask(masks: np.ndarray, dets: list) -> np.ndarray:
    """Remap instance-id mask (index into det list) -> class label."""
    lut = np.array([int(det.label) for det in dets], dtype=np.int16)

    semantic = np.full(masks.shape, 255, dtype=np.int16)
    valid = (masks < 255) & (masks < lut.size)
    if np.any(valid):
        semantic[valid] = lut[masks[valid]]
    return semantic


def test_building(converter: InstanceToSemanticMask):
    converter.build(OutputMock())


def test_wrong_input_type(converter: InstanceToSemanticMask):
    with pytest.raises(TypeError):
        converter.process(dai.ImgFrame())


def test_processing_instance_to_semantic(
    converter: InstanceToSemanticMask, duration: float
):
    o = OutputMock()
    converter.build(o)

    q_in = o.createOutputQueue()
    q_out = converter.out.createOutputQueue()

    # base message with detections
    msg = create_img_detections_extended()
    n = len(msg.detections)

    # If there are no detections in the test fixture, the mapping isn't meaningful.
    # In that case, just ensure it doesn't crash and passes through.
    if n == 0:
        msg.setCvSegmentationMask(np.array([[0, 1], [2, 3]], dtype=np.uint8))
        q_in.send(msg)
        converter.process(q_in.get())
        out = q_out.get()
        assert isinstance(out, ImgDetectionsExtended)
        np.testing.assert_array_equal(out.masks, msg.masks)
        return

    # Build a deterministic instance-id mask:
    # - valid ids: 0..min(n-1, 2)
    # - invalid ids: -1, n (out of range)
    h, w = 16, 16
    masks = np.full((h, w), 255, dtype=np.int16)
    masks[0:8, 0:8] = 0
    if n > 1:
        masks[0:8, 8:16] = 1
    if n > 2:
        masks[8:16, 0:8] = 2
    masks[8:16, 8:16] = n  # out of range -> should become -1

    msg.setCvSegmentationMask(masks.astype(np.uint8))
    semantic_mask = _instance_to_semantic_mask(masks, msg.detections)

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time() - start_time:.1f}s elapsed, "
                f"{duration - time.time() + start_time:.1f}s remaining"
            )
            last_log_time = time.time()

        q_in.send(msg)
        converter.process(q_in.get())
        out = q_out.get()

        assert isinstance(out, dai.ImgDetections)
        assert len(out.detections) == n
        np.testing.assert_array_equal(out.getCvSegmentationMask(), semantic_mask)

        # sanity: out-of-range region should be 255
        assert (out.getCvSegmentationMask()[8:16, 8:16] == 255).all()
