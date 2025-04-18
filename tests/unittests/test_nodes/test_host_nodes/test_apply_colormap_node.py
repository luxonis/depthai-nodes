import time

import cv2
import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import ApplyColormap
from tests.utils import OutputMock

from .utils.create_message import (
    ARRAYS,
    create_img_detections_extended,
    create_img_frame,
    create_map,
)

ARR = ARRAYS["2d"]
HEIGHT, WIDTH = ARR.shape
MAX_VALUE = ARR.max().item()


def make_colormap(colormap_value: int) -> np.ndarray:
    colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), colormap_value)
    colormap[0] = [0, 0, 0]  # Set zero values to black
    return colormap


def apply_colormap(arr: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(
        ((arr / arr.max()) * 255).astype(np.uint8),
        colormap,
    )


@pytest.fixture(scope="session")
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def colorizer():
    return ApplyColormap()


def test_initialization(colorizer: ApplyColormap):
    assert colorizer.parentInitialized()


def test_building(colorizer: ApplyColormap):
    colorizer.build(OutputMock())


def test_parameter_setting(
    colorizer: ApplyColormap,
    colormap_value: int = cv2.COLORMAP_WINTER,
    max_value: int = MAX_VALUE,
):
    # colormap_value
    colorizer.setColormap(colormap_value)
    assert np.array_equal(colorizer._colormap, make_colormap(colormap_value))
    # test isinstance(colormap_value, int)
    with pytest.raises(ValueError):
        colorizer.setColormap("not an integer")

    # max_value
    colorizer.setMaxValue(max_value)
    assert colorizer._max_value == max_value
    # test isinstance(max_value, int)
    with pytest.raises(ValueError):
        colorizer.setMaxValue("not an integer")


@pytest.mark.parametrize(
    "colormap_value", [cv2.COLORMAP_HOT, cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO]
)
def test_processing(
    colorizer: ApplyColormap, colormap_value: int, duration: int = 1e-6
):
    o_array = OutputMock()
    colorizer.build(o_array)
    colorizer.setColormap(colormap_value)

    q_arr = o_array.createOutputQueue()
    q_colorizer = colorizer.out.createOutputQueue()

    for arr in [
        create_img_frame(
            image=ARR[..., np.newaxis], img_frame_type=dai.ImgFrame.Type.RAW8
        ),  # dai.ImgFrame
        create_map(ARR.astype(np.float32)),  # Map2D
        create_img_detections_extended(masks=ARR),  # ImgDetectionsExtended
    ]:
        start_time = time.time()
        while time.time() - start_time < duration:
            q_arr.send(arr)
            colorizer.process(q_arr.get())
            arr_colored = q_colorizer.get()

            assert isinstance(arr_colored, dai.ImgFrame)
            assert arr_colored.getCvFrame().shape == (HEIGHT, WIDTH, 3)
            assert np.array_equal(
                arr_colored.getCvFrame(),
                apply_colormap(ARR, make_colormap(colormap_value)),
            )
