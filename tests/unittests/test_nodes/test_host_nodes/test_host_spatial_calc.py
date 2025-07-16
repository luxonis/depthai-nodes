import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node.host_spatials_calc import HostSpatialsCalc

from .utils.calibration_handler import get_calibration_handler


@pytest.fixture
def calib_data():
    return get_calibration_handler()


@pytest.fixture
def depth_frame():
    frame = np.random.randint(200, 30000, (720, 1280), dtype=np.uint16)
    img_frame = dai.ImgFrame()
    img_frame.setCvFrame(frame, dai.ImgFrame.Type.RAW16)
    img_frame.setWidth(1280)
    img_frame.setHeight(720)
    return img_frame


@pytest.fixture
def host_spatials_calc(calib_data):
    return HostSpatialsCalc(calib_data)


def test_set_lower_threshold(host_spatials_calc: HostSpatialsCalc):
    host_spatials_calc.setLowerThreshold(100)
    assert host_spatials_calc.thresh_low == 100

    host_spatials_calc.setLowerThreshold(150.5)
    assert host_spatials_calc.thresh_low == 150

    with pytest.raises(TypeError):
        host_spatials_calc.setLowerThreshold("invalid")


def test_set_upper_threshold(host_spatials_calc: HostSpatialsCalc):
    host_spatials_calc.setUpperThreshold(50000)
    assert host_spatials_calc.thresh_high == 50000

    host_spatials_calc.setUpperThreshold(25000.5)
    assert host_spatials_calc.thresh_high == 25000

    with pytest.raises(TypeError):
        host_spatials_calc.setUpperThreshold("invalid")


def test_set_delta_roi(host_spatials_calc: HostSpatialsCalc):
    host_spatials_calc.setDeltaRoi(10)
    assert host_spatials_calc.delta == 10

    host_spatials_calc.setDeltaRoi(15.5)
    assert host_spatials_calc.delta == 15

    with pytest.raises(TypeError):
        host_spatials_calc.setDeltaRoi("invalid")


def test_check_input_roi(
    host_spatials_calc: HostSpatialsCalc, depth_frame: dai.ImgFrame
):
    roi = [100, 100, 200, 200]
    result = host_spatials_calc._check_input(roi, depth_frame.getFrame())
    assert result == roi


def test_check_input_point(
    host_spatials_calc: HostSpatialsCalc, depth_frame: dai.ImgFrame
):
    point = [100, 100]
    result = host_spatials_calc._check_input(point, depth_frame.getFrame())
    assert result == [95, 95, 105, 105]


def test_check_input_invalid(
    host_spatials_calc: HostSpatialsCalc, depth_frame: dai.ImgFrame
):
    with pytest.raises(ValueError):
        host_spatials_calc._check_input([100], depth_frame.getFrame())


def test_calc_spatials(host_spatials_calc: HostSpatialsCalc, depth_frame: dai.ImgFrame):
    roi = [100, 100, 200, 200]
    spatials = host_spatials_calc.calc_spatials(depth_frame, roi)
    assert "x" in spatials
    assert "y" in spatials
    assert "z" in spatials
    assert isinstance(spatials["x"], float)
    assert isinstance(spatials["y"], float)
    assert isinstance(spatials["z"], float)
