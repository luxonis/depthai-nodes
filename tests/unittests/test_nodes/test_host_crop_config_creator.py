import depthai as dai
import numpy as np
import pytest
import yaml
from conftest import Output
from pytest import FixtureRequest

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.node import CropConfigsCreatorNode


@pytest.fixture
def crop_configs_creator():
    return CropConfigsCreatorNode()


@pytest.fixture
def empty_img_detections_extended():
    detections = ImgDetectionsExtended()

    return detections


@pytest.fixture
def single_img_detections_extended():
    detections = ImgDetectionsExtended()
    detection = ImgDetectionExtended()
    xmin = 0.3
    xmax = 0.5
    ymin = 0.3
    ymax = 0.5
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    detection.rotated_rect = (x_center, y_center, width, height, 0)
    detection.rotated_rect.angle = 0
    detection.label = 1
    detection.confidence = 0.9
    detections.detections = [detection]

    return detections


@pytest.fixture
def img_detections_extended():
    detections = ImgDetectionsExtended()

    c_min = np.linspace(0.1, 0.8, 8)
    c_max = np.linspace(0.2, 0.9, 8)
    for xmin, xmax, ymin, ymax in zip(c_min, c_max, c_min, c_max):
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        detection = ImgDetectionExtended()
        detection.rotated_rect = (x_center, y_center, width, height, 0)
        detection.label = 1
        detection.confidence = 0.9
        detections.detections.append(detection)

    return detections


@pytest.mark.parametrize(
    "detections",
    [
        "empty_img_detections_extended",
        "single_img_detections_extended",
        "img_detections_extended",
    ],
)
def test_img_detections(
    crop_configs_creator: CropConfigsCreatorNode,
    request: FixtureRequest,
    detections: str,
):
    img_detections: ImgDetectionsExtended = request.getfixturevalue(detections)
    img_detections_msg = Output()

    crop_configs_creator.build(
        detections_input=img_detections_msg, w=1000, h=1000, target_w=640, target_h=480
    )

    assert crop_configs_creator.w == 1000
    assert crop_configs_creator.h == 1000
    assert crop_configs_creator.target_w == 640
    assert crop_configs_creator.target_h == 480
    assert crop_configs_creator.n_detections == 100

    q_img_detections_msg = img_detections_msg.createOutputQueue()

    q_configs = crop_configs_creator.config_output.createOutputQueue()
    q_detections = crop_configs_creator.detections_output.createOutputQueue()

    img_detections_msg.send(img_detections)

    crop_configs_creator.process(q_img_detections_msg.get())

    skip_config: dai.ImageManipConfigV2 = q_configs.get()
    config_str = str(skip_config)
    config_dict = yaml.safe_load(config_str)

    assert config_dict["skipCurrentImage"]

    crop_detections: ImgDetectionsExtended = q_detections.get()

    for det in img_detections.detections:
        config_msg = q_configs.get()

        config_str = str(config_msg)
        config_str = config_str.replace("[", "")
        config_str = config_str.replace("]", "")
        print(config_str)
        config_dict = yaml.safe_load(config_str)

        assert not config_dict["skipCurrentImage"]
        assert config_dict["reusePreviousImage"]
        assert config_dict["base"]["outputHeight"] == 480
        assert config_dict["base"]["outputWidth"] == 640

        true_coords = det.rotated_rect.getOuterRect()
        return_coords = crop_detections.rotated_rect.getOuterRect()

        assert np.allclose(true_coords, return_coords, atol=1e-8)
        assert det.label == crop_detections.label
        assert det.confidence == crop_detections.confidence
