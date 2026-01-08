import time

import cv2
import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import ApplyDepthColormap
from tests.utils import (
    ARRAYS,
    LOG_INTERVAL,
    OutputMock,
)

ARR = ARRAYS["2d"]
HEIGHT, WIDTH = ARR.shape

# Depth array with guaranteed invalid zeros in a specific region
DEPTH_ARR = ARR.astype(np.uint16) + 1  # valid depth
DEPTH_ARR[:10, :10] = 0  # invalid region

# Your parameter lists
colormap_values = [cv2.COLORMAP_HOT, cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO]
percentile_ranges = [(2.0, 98.0), (0.0, 100.0), (10.0, 90.0)]


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


def create_depth_frame(depth: np.ndarray) -> dai.ImgFrame:
    assert depth.ndim == 2
    img = dai.ImgFrame()
    img.setCvFrame(depth, dai.ImgFrame.Type.RAW16)
    img.setWidth(depth.shape[1])
    img.setHeight(depth.shape[0])
    img.setType(dai.ImgFrame.Type.RAW16)
    return img


def apply_depth_colormap(
    depth: np.ndarray,
    colormap: np.ndarray,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    invalid = depth <= 0
    valid = depth[~invalid]
    if valid.size == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    low = float(np.percentile(valid, p_low))
    high = float(np.percentile(valid, p_high))

    if high <= low or low <= 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    depth_f = depth.astype(np.float32, copy=False)
    scaled = (depth_f - low) / (high - low) * 255.0
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    scaled[invalid] = 0

    colored = cv2.applyColorMap(scaled, colormap)
    colored[invalid] = 0
    return colored


@pytest.fixture
def depth_colorizer():
    return ApplyDepthColormap()


def test_building(depth_colorizer: ApplyDepthColormap):
    depth_colorizer.build(OutputMock())


def test_parameter_setting(
    depth_colorizer: ApplyDepthColormap,
    colormap_value: int = cv2.COLORMAP_WINTER,
):
    # colormap_value
    depth_colorizer.setColormap(colormap_value)
    assert np.array_equal(depth_colorizer._colormap, make_colormap(colormap_value))
    # test isinstance(colormap_value, int)
    with pytest.raises(ValueError):
        depth_colorizer.setColormap("not an integer")
    # custom colormap
    custom_colormap = np.random.randint(0, 256, size=(256, 1, 3), dtype=np.uint8)
    depth_colorizer.setColormap(custom_colormap)
    # test wrong shape
    with pytest.raises(ValueError):
        depth_colorizer.setColormap(
            np.random.randint(0, 256, size=(255, 1, 3), dtype=np.uint8)
        )
    with pytest.raises(ValueError):
        depth_colorizer.setColormap(
            np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)
        )
    # test wrong dtype
    with pytest.raises(ValueError):
        depth_colorizer.setColormap(np.random.rand(256, 1, 3).astype(np.float32))

    # percentile range
    depth_colorizer.setPercentileRange(2.0, 98.0)
    assert depth_colorizer._p_low == 2.0
    assert depth_colorizer._p_high == 98.0

    with pytest.raises(ValueError):
        depth_colorizer.setPercentileRange(-1.0, 98.0)
    with pytest.raises(ValueError):
        depth_colorizer.setPercentileRange(50.0, 50.0)
    with pytest.raises(ValueError):
        depth_colorizer.setPercentileRange(98.0, 2.0)
    with pytest.raises(ValueError):
        depth_colorizer.setPercentileRange(0.0, 101.0)


@pytest.mark.parametrize("colormap_value", colormap_values)
@pytest.mark.parametrize("p_low,p_high", percentile_ranges)
def test_processing(
    depth_colorizer: ApplyDepthColormap,
    colormap_value: int,
    p_low: float,
    p_high: float,
    duration: float,
):
    total_combinations = len(colormap_values) * len(percentile_ranges)
    modified_duration = duration / total_combinations

    o_depth = OutputMock()
    depth_colorizer.build(o_depth)
    depth_colorizer.setColormap(colormap_value)
    depth_colorizer.setPercentileRange(p_low, p_high)

    q_depth = o_depth.createOutputQueue()
    q_out = depth_colorizer.out.createOutputQueue()

    depth_msg = create_depth_frame(DEPTH_ARR)

    depth_colormap = apply_depth_colormap(
        DEPTH_ARR,
        make_colormap(colormap_value),
        p_low,
        p_high,
    )

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < modified_duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time() - start_time:.1f}s elapsed, "
                f"{modified_duration - time.time() + start_time:.1f}s remaining"
            )
            last_log_time = time.time()

        q_depth.send(depth_msg)
        depth_colorizer.process(q_depth.get())
        out = q_out.get()

        assert isinstance(out, dai.ImgFrame)
        out_frame = out.getCvFrame()
        assert out_frame.shape == (HEIGHT, WIDTH, 3)

        # correctness
        assert np.array_equal(out_frame, depth_colormap)

        # invalid region should be black
        assert (out_frame[:10, :10] == 0).all()
