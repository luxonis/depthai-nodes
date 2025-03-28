import depthai as dai
import numpy as np
import pytest
from conftest import Output
from utils.create_message import create_img_frame

from depthai_nodes.node import ImgFrameOverlay

HEIGHT, WIDTH = 5, 5
WEIGHT = 0.5
IMG1 = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
IMG2 = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
IMG3 = np.random.randint(0, 255, (HEIGHT, WIDTH * 2, 3), dtype=np.uint8)


def test_initialization():
    overlayer = ImgFrameOverlay(background_weight=WEIGHT)
    assert overlayer._background_weight == WEIGHT


def test_building():
    ImgFrameOverlay().build(Output(), Output())


def test_parameter_setting():
    overlayer = ImgFrameOverlay()
    overlayer.SetBackgroundWeight(WEIGHT)
    assert overlayer._background_weight == WEIGHT

    # test isinstance(background_weight, float)
    with pytest.raises(ValueError):
        overlayer.SetBackgroundWeight("not a float")

    # test 0.0 <= background_weight <= 1.0
    with pytest.raises(ValueError):
        overlayer.SetBackgroundWeight(WEIGHT * 100)


def test_processing():
    img_frame1: dai.ImgFrame = create_img_frame(IMG1)
    img_frame2: dai.ImgFrame = create_img_frame(IMG2)
    img_frame3: dai.ImgFrame = create_img_frame(IMG3)

    o_background = Output()
    o_foreground = Output()
    overlayer = ImgFrameOverlay().build(o_background, o_foreground)
    overlayer.SetBackgroundWeight(WEIGHT)

    q_background = o_background.createOutputQueue()
    q_foreground = o_foreground.createOutputQueue()
    q_overlayer = overlayer.out.createOutputQueue()

    q_background.send(img_frame1)
    q_foreground.send(img_frame2)
    overlayer.process(q_background.get(), q_foreground.get())
    overlay = q_overlayer.get()

    assert isinstance(overlay, dai.ImgFrame)
    assert np.array_equal(
        overlay.getCvFrame(),
        np.round(IMG1 * WEIGHT + IMG2 * (1 - WEIGHT)).astype(np.uint8),
    )

    q_background.send(img_frame1)
    q_foreground.send(img_frame3)
    overlayer.process(q_background.get(), q_foreground.get())
    overlay = q_overlayer.get()

    assert overlay.getCvFrame().shape == img_frame1.getCvFrame().shape
