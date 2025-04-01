import depthai as dai
import numpy as np
import pytest
import time

from depthai_nodes.node import ImgFrameOverlay
from utils.create_message import create_img_frame
from conftest import Output

HEIGHT, WIDTH = 5, 5
ALPHA = 0.5
FRAME1 = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
FRAME2 = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
FRAME3 = np.random.randint(0, 255, (HEIGHT, WIDTH * 2, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def duration(request):
    return request.config.getoption("--duration")


def test_initialization():
    overlayer = ImgFrameOverlay(alpha=ALPHA)
    assert overlayer._alpha == ALPHA


def test_building():
    overlayer = ImgFrameOverlay().build(Output(), Output(), alpha=ALPHA)
    assert overlayer._alpha == ALPHA


def test_parameter_setting():
    overlayer = ImgFrameOverlay()
    overlayer.SetAlpha(ALPHA)
    assert overlayer._alpha == ALPHA

    # test isinstance(alpha, float)
    with pytest.raises(ValueError):
        overlayer.SetAlpha("not a float")

    # test 0.0 <= alpha <= 1.0
    with pytest.raises(ValueError):
        overlayer.SetAlpha(ALPHA * 100)


def test_processing(
    duration: int = 1e-6,
):
    img_frame1: dai.ImgFrame = create_img_frame(FRAME1)
    img_frame2: dai.ImgFrame = create_img_frame(FRAME2)
    img_frame3: dai.ImgFrame = create_img_frame(FRAME3)

    o_background = Output()
    o_foreground = Output()
    overlayer = ImgFrameOverlay().build(o_background, o_foreground)
    overlayer.SetAlpha(ALPHA)

    q_background = o_background.createOutputQueue()
    q_foreground = o_foreground.createOutputQueue()
    q_overlayer = overlayer.out.createOutputQueue()

    start_time = time.time()
    while time.time() - start_time < duration:
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
