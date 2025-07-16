import time

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import ImgFrameOverlay
from tests.utils import LOG_INTERVAL, OutputMock, create_img_frame

HEIGHT, WIDTH = 5, 5
ALPHA = 0.5
FRAME1 = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
FRAME2 = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
FRAME3 = np.random.randint(0, 255, (HEIGHT, WIDTH * 2, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def duration(request):
    d = request.config.getoption("--duration")
    if d is None:
        return 1e-6
    return d


@pytest.fixture
def overlayer():
    return ImgFrameOverlay()


def test_building(overlayer: ImgFrameOverlay):
    overlayer.build(OutputMock(), OutputMock(), alpha=ALPHA)
    assert overlayer._alpha == ALPHA


def test_parameter_setting(overlayer: ImgFrameOverlay):
    overlayer.setAlpha(ALPHA)
    assert overlayer._alpha == ALPHA

    # test isinstance(alpha, float)
    with pytest.raises(ValueError):
        overlayer.setAlpha("not a float")

    # test 0.0 <= alpha <= 1.0
    with pytest.raises(ValueError):
        overlayer.setAlpha(ALPHA * 100)


def test_processing(
    overlayer: ImgFrameOverlay,
    duration: float,
):
    img_frame1: dai.ImgFrame = create_img_frame(FRAME1)
    img_frame2: dai.ImgFrame = create_img_frame(FRAME2)
    img_frame3: dai.ImgFrame = create_img_frame(FRAME3)

    o_background = OutputMock()
    o_foreground = OutputMock()
    overlayer.build(o_background, o_foreground)
    overlayer.setAlpha(ALPHA)

    q_background = o_background.createOutputQueue()
    q_foreground = o_foreground.createOutputQueue()
    q_overlayer = overlayer.out.createOutputQueue()

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time()-start_time:.1f}s elapsed, {duration-time.time()+start_time:.1f}s remaining"
            )
            last_log_time = time.time()
        q_background.send(img_frame1)
        q_foreground.send(img_frame2)
        overlayer.process(q_background.get(), q_foreground.get())
        overlay = q_overlayer.get()

        assert isinstance(overlay, dai.ImgFrame)
        assert np.array_equal(
            overlay.getCvFrame(),
            np.round(FRAME1 * ALPHA + FRAME2 * (1 - ALPHA)).astype(np.uint8),
        )

        q_background.send(img_frame1)
        q_foreground.send(img_frame3)
        overlayer.process(q_background.get(), q_foreground.get())
        overlay = q_overlayer.get()

        assert overlay.getCvFrame().shape == img_frame1.getCvFrame().shape
