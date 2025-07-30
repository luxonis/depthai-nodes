import time
from typing import Callable

import cv2
import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import ApplyColormap
from tests.utils import (
    ARRAYS,
    LOG_INTERVAL,
    OutputMock,
    create_img_detections_extended,
    create_img_frame,
    create_map,
)

ARR = ARRAYS["2d"]
HEIGHT, WIDTH = ARR.shape
MAX_VALUE = ARR.max().item()

# Your parameter lists
colormap_values = [cv2.COLORMAP_HOT, cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO]
arr_creators = [
    lambda: create_img_frame(
        image=ARR[..., np.newaxis], img_frame_type=dai.ImgFrame.Type.RAW8
    ),
    lambda: create_map(ARR.astype(np.float32)),
    lambda: create_img_detections_extended(masks=ARR),
]


@pytest.fixture(scope="session")
def duration(request):
    d = request.config.getoption("--duration")
    if d is None:
        return 1e-6
    return d


def make_colormap(colormap_value: int) -> np.ndarray:
    colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), colormap_value)
    colormap[0] = [0, 0, 0]  # Set zero values to black
    return colormap


def apply_colormap(arr: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(
        ((arr / arr.max()) * 255).astype(np.uint8),
        colormap,
    )


@pytest.fixture
def colorizer():
    return ApplyColormap()


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
    # custom colormap
    custom_colormap = np.random.randint(0, 256, size=(256, 1, 3), dtype=np.uint8)
    colorizer.setColormap(custom_colormap)
    # test wrong shape
    with pytest.raises(ValueError):
        colorizer.setColormap(
            np.random.randint(0, 256, size=(255, 1, 3), dtype=np.uint8)
        )
    with pytest.raises(ValueError):
        colorizer.setColormap(np.random.randint(0, 256, size=(256, 3), dtype=np.uint8))
    # test wrong dtype
    with pytest.raises(ValueError):
        colorizer.setColormap(np.random.rand(256, 1, 3).astype(np.float32))

    # max_value
    colorizer.setMaxValue(max_value)
    assert colorizer._max_value == max_value
    # test isinstance(max_value, int)
    with pytest.raises(ValueError):
        colorizer.setMaxValue("not an integer")


@pytest.mark.parametrize("colormap_value", colormap_values)
@pytest.mark.parametrize("arr_creator", arr_creators)
def test_processing(
    colorizer: ApplyColormap,
    colormap_value: int,
    arr_creator: Callable,
    duration: float,
):
    total_combinations = len(colormap_values) * len(arr_creators)

    modified_duration = duration / total_combinations
    o_array = OutputMock()
    colorizer.build(o_array)
    colorizer.setColormap(colormap_value)

    q_arr = o_array.createOutputQueue()
    q_colorizer = colorizer.out.createOutputQueue()

    arr = arr_creator()
    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < modified_duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time() - start_time:.1f}s elapsed, {modified_duration - time.time() + start_time:.1f}s remaining"
            )
            last_log_time = time.time()
        q_arr.send(arr)
        colorizer.process(q_arr.get())
        arr_colored = q_colorizer.get()

        assert isinstance(arr_colored, dai.ImgFrame)
        assert arr_colored.getCvFrame().shape == (HEIGHT, WIDTH, 3)
        assert np.array_equal(
            arr_colored.getCvFrame(),
            apply_colormap(ARR, make_colormap(colormap_value)),
        )
