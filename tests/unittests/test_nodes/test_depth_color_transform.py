import time

import cv2
import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import DepthColorTransform


@pytest.fixture(scope="session")
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def disp_array():
    return np.array([[1], [2], [3]], dtype=np.uint8)


@pytest.fixture
def disp_shape(disp_array):
    return disp_array.shape


@pytest.fixture
def disp_frame(disp_array):
    frame = dai.ImgFrame()
    frame.setCvFrame(disp_array, dai.ImgFrame.Type.RAW8)
    return frame


@pytest.fixture
def empty_frame(disp_shape):
    frame = dai.ImgFrame()
    frame.setCvFrame(np.zeros(disp_shape), dai.ImgFrame.Type.RAW8)
    return frame


def test_dimensions_type(disp_frame, disp_shape, duration):
    color_transform = DepthColorTransform()
    q = color_transform.out.createOutputQueue()
    color_transform.process(disp_frame)
    output = q.get()
    assert isinstance(output, dai.ImgFrame)
    assert output.getCvFrame().shape == (*disp_shape, 3)

    if duration:
        start_time = time.time()

        while time.time() - start_time < duration:
            color_transform.process(disp_frame)
            output = q.get()
            assert isinstance(output, dai.ImgFrame)
            assert output.getCvFrame().shape == (*disp_shape, 3)


def test_empty_frame(empty_frame, duration):
    color_transform = DepthColorTransform()
    q = color_transform.out.createOutputQueue()
    color_transform.process(empty_frame)
    output = q.get()
    assert isinstance(output, dai.ImgFrame)
    output_matrix = output.getCvFrame()
    assert np.all(output_matrix == 0)

    if duration:
        start_time = time.time()

        while time.time() - start_time < duration:
            color_transform.process(empty_frame)
            output = q.get()
            assert isinstance(output, dai.ImgFrame)
            output_matrix = output.getCvFrame()
            assert np.all(output_matrix == 0)


@pytest.mark.parametrize(
    "color_map", [cv2.COLORMAP_HOT, cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO]
)
def test_color_map_applied(disp_array, disp_frame, color_map, duration):
    color_transform = DepthColorTransform()
    color_transform.setColormap(color_map)
    factor = 2
    maximum_disp = disp_array.max() * factor
    color_transform.setMaxDisparity(maximum_disp)
    q = color_transform.out.createOutputQueue()
    color_transform.process(disp_frame)
    output = q.get()
    assert isinstance(output, dai.ImgFrame)
    output_matrix = output.getCvFrame()
    factored_disp = (disp_array / maximum_disp * 255).astype(np.uint8)
    assert np.all(
        output_matrix == cv2.applyColorMap(factored_disp, color_transform._colormap)
    )

    if duration:
        start_time = time.time()

        while time.time() - start_time < duration:
            color_transform.process(disp_frame)
            output = q.get()
            assert isinstance(output, dai.ImgFrame)
            output_matrix = output.getCvFrame()
            assert np.all(
                output_matrix
                == cv2.applyColorMap(factored_disp, color_transform._colormap)
            )
