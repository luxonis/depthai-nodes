import time
from typing import Callable, Dict, Optional, Tuple, Union

import depthai as dai
import numpy as np
import pytest
import yaml
from conftest import Output
from pytest import FixtureRequest

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.node import CropConfigsCreator


@pytest.fixture(scope="session")
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def crop_configs_creator():
    return CropConfigsCreator()


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
    return create_img_detections_extended()


@pytest.fixture
def img_detections_with_map():
    detections = create_img_detections_extended()
    np.random.seed(1)
    map = np.random.randint(-1, 10, (1000, 1000), dtype=np.int16)

    detections.masks = map

    return detections


def create_img_detections_extended():
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


def img_detections_extended_identity_map(detections: ImgDetectionsExtended):
    return detections


def cast_to_dai_img_detections(
    img_detections_extended: ImgDetectionsExtended,
) -> dai.ImgDetections:
    img_detections = dai.ImgDetections()
    converted_detections = []

    for det in img_detections_extended.detections:
        converted_det = dai.ImgDetection()
        converted_det.label = det.label
        converted_det.confidence = det.confidence
        xmin, ymin, xmax, ymax = det.rotated_rect.getOuterRect()
        converted_det.xmin = xmin
        converted_det.ymin = ymin
        converted_det.xmax = xmax
        converted_det.ymax = ymax
        converted_detections.append(converted_det)

    img_detections.detections = converted_detections
    img_detections.setSequenceNum(img_detections_extended.getSequenceNum())
    img_detections.setTimestamp(img_detections_extended.getTimestamp())
    img_detections.setTransformation(img_detections_extended.transformation)

    return img_detections


def verify_resized_crop(crop_config: Dict, target_width: int, target_height: int):
    assert not crop_config["skipCurrentImage"]
    assert crop_config["reusePreviousImage"]
    assert np.isclose(crop_config["base"]["outputHeight"], target_height)
    assert np.isclose(crop_config["base"]["outputWidth"], target_width)


def verify_output_detection(
    gt_detection: ImgDetectionExtended, return_detection: ImgDetectionExtended
):
    gt_coords = gt_detection.rotated_rect.getOuterRect()
    return_coords = return_detection.rotated_rect.getOuterRect()

    assert np.allclose(gt_coords, return_coords, atol=1e-6)
    assert return_detection.label == gt_detection.label
    assert np.isclose(return_detection.confidence, gt_detection.confidence, atol=1e-6)
    assert np.isclose(
        return_detection.rotated_rect.angle, gt_detection.rotated_rect.angle, atol=1e-6
    )


def verify_crop_det_assertions(
    crop_detections: ImgDetectionsExtended, img_detections: ImgDetectionsExtended
):
    assert crop_detections.getSequenceNum() == img_detections.getSequenceNum()
    assert crop_detections.getTimestamp() == img_detections.getTimestamp()
    assert crop_detections.transformation == img_detections.transformation


def load_crop_dict(config_msg) -> Dict:
    config_str = str(config_msg)
    config_str = config_str.replace("[", "")
    config_str = config_str.replace("]", "")
    config_dict = yaml.safe_load(config_str)

    return config_dict


@pytest.mark.parametrize(
    "detections",
    [
        "empty_img_detections_extended",
        "single_img_detections_extended",
        "img_detections_extended",
        "img_detections_with_map",
    ],
)
@pytest.mark.parametrize(
    "target_size",
    [
        None,
        (640, 480),
        (1280, 720),
    ],
)
@pytest.mark.parametrize(
    "preprocess_function",
    [
        img_detections_extended_identity_map,
        cast_to_dai_img_detections,
    ],
)
def test_img_detections(
    duration: int,
    crop_configs_creator: CropConfigsCreator,
    request: FixtureRequest,
    detections: str,
    target_size: Optional[Tuple[int, int]],
    preprocess_function: Callable[
        [ImgDetectionsExtended], Union[ImgDetectionsExtended, dai.ImgDetections]
    ],
):
    source_size = (1000, 1000)
    img_detections: ImgDetectionsExtended = request.getfixturevalue(detections)

    img_detections_msg = Output()
    crop_configs_creator.build(
        detections_input=img_detections_msg,
        source_size=source_size,
        target_size=target_size,
    )

    assert crop_configs_creator.w == source_size[0]
    assert crop_configs_creator.h == source_size[1]
    if target_size is not None:
        assert crop_configs_creator.target_w == target_size[0]
        assert crop_configs_creator.target_h == target_size[1]
    assert crop_configs_creator.resize_mode == dai.ImageManipConfigV2.ResizeMode.STRETCH

    q_img_detections_msg = img_detections_msg.createOutputQueue()

    q_configs = crop_configs_creator.config_output.createOutputQueue()
    q_detections = crop_configs_creator.detections_output.createOutputQueue()

    img_detections_msg.send(preprocess_function(img_detections))
    crop_configs_creator.process(q_img_detections_msg.get())

    skip_config: dai.ImageManipConfigV2 = q_configs.get()
    config_str = str(skip_config)
    config_dict = yaml.safe_load(config_str)
    assert config_dict["skipCurrentImage"]

    crop_detections: ImgDetectionsExtended = q_detections.get()
    verify_crop_det_assertions(crop_detections, img_detections)

    for true_det, crop_det in zip(
        img_detections.detections, crop_detections.detections
    ):
        config_msg = q_configs.get()
        config_dict = load_crop_dict(config_msg)

        verify_output_detection(true_det, crop_det)
        if target_size is not None:
            verify_resized_crop(config_dict, target_size[0], target_size[1])

    if duration:
        start_time = time.time()

        while time.time() - start_time < duration:
            img_detections_msg.send(preprocess_function(img_detections))
            crop_configs_creator.process(q_img_detections_msg.get())
            skip_config: dai.ImageManipConfigV2 = q_configs.get()
            config_str = str(skip_config)
            config_dict = yaml.safe_load(config_str)
            assert config_dict["skipCurrentImage"]

            crop_detections: ImgDetectionsExtended = q_detections.get()
            verify_crop_det_assertions(crop_detections, img_detections)

            for true_det, crop_det in zip(
                img_detections.detections, crop_detections.detections
            ):
                config_msg = q_configs.get()
                config_dict = load_crop_dict(config_msg)

                verify_output_detection(true_det, crop_det)
                if target_size is not None:
                    verify_resized_crop(config_dict, target_size[0], target_size[1])
