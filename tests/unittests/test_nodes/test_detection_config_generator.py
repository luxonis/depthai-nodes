from typing import List, Optional, Union

import depthai as dai
import pytest

from depthai_nodes.node.utils import generate_script_content


@pytest.fixture
def resize_width():
    return 256


@pytest.fixture
def resize_height():
    return 256


def test_rvc3_unsupported(resize_width, resize_height):
    with pytest.raises(ValueError, match="Unsupported"):
        generate_script_content("rvc3", resize_width, resize_height)


class ImageManipConfigV2(dai.ImageManipConfigV2):
    def __init__(self):
        super().__init__()
        self._output_size: Optional[tuple[int, int]] = None
        self._crop_rotated_rect: Optional[dai.RotatedRect] = None

    def setOutputSize(
        self, w, h, mode: Optional[dai.ImageManipConfigV2.ResizeMode] = None
    ):
        self._output_size = w, h
        if mode:
            return super().setOutputSize(w, h, mode)
        else:
            return super().setOutputSize(w, h)

    def getOutputSize(self):
        return self._output_size

    def addCropRotatedRect(self, rect: dai.RotatedRect, normalizedCoords: bool):
        self._crop_rotated_rect = rect
        super().addCropRotatedRect(rect, normalizedCoords)

    def getCropRotatedRect(self):
        return self._crop_rotated_rect


class Frame:
    def __init__(self, sequence_num: int):
        self.sequence_num = sequence_num


class Node:
    INPUT_FRAMES_KEY = "preview"
    INPUT_DETECTIONS_KEY = "det_in"
    OUTPUT_CONFIG_KEY = "manip_cfg"
    OUTPUT_FRAMES_KEY = "manip_img"

    class Input:
        def __init__(self, items: List):
            self._items = items

        def get(self):
            return self._items.pop(0)

        def empty(self):
            return not any(self._items)

        @property
        def items(self):
            return self._items

    class Output:
        def __init__(self):
            self._items = []

        def send(self, item):
            self._items.append(item)

        @property
        def items(self):
            return self._items

    def __init__(
        self,
        preview: Input,
        detections: Input,
        manip_config: Output,
        manip_image: Output,
    ):
        self.inputs = {
            self.INPUT_FRAMES_KEY: preview,
            self.INPUT_DETECTIONS_KEY: detections,
        }
        self.outputs = {
            self.OUTPUT_CONFIG_KEY: manip_config,
            self.OUTPUT_FRAMES_KEY: manip_image,
        }

    def warn(self, msg: str):
        raise Warning(msg)


def create_node(preview: List[Frame], detections: List[dai.ImgDetections]):
    return Node(
        preview=Node.Input(preview),
        detections=Node.Input(detections),
        manip_config=Node.Output(),
        manip_image=Node.Output(),
    )


@pytest.fixture(
    params=[
        [
            [(0, 0, 0.5, 0.1, 0.4, 0.7)],  # first detection set
            [
                (1, 0.2, 0.7, 0.3, 0.6, 0.8),
                (2, 0.1, 0.4, 0.2, 0.5, 0.9),
            ],  # second set with 2 detections
            [
                (3, 0.4, 0.9, 0.1, 0.8, 0.6),
                (4, 0.3, 0.8, 0.2, 0.7, 0.75),
                (5, 0.1, 0.6, 0.3, 0.9, 0.85),
            ],  # third set with 3 detections
        ]
    ]
)
def detections(request):
    detections_params = request.param
    detections_list: List[dai.ImgDetections] = []
    for detection_param in detections_params:
        detections: List[dai.ImgDetection] = []
        for label, ymin, ymax, xmin, xmax, conf in detection_param:
            detection = dai.ImgDetection()
            detection.label = label
            detection.ymin = ymin
            detection.ymax = ymax
            detection.xmin = xmin
            detection.xmax = xmax
            detection.confidence = conf
            detections.append(detection)
        img_detections = dai.ImgDetections()
        img_detections.detections = detections
        detections_list.append(img_detections)
    return detections_list


@pytest.fixture
def frames(detections: List[dai.ImgDetections]):
    return [Frame(i) for i, _ in enumerate(detections)]


@pytest.fixture
def node(frames, detections):
    return create_node(frames, detections)


@pytest.fixture
def node_input_frames(node) -> List[Frame]:
    return node.inputs[Node.INPUT_FRAMES_KEY].items


@pytest.fixture
def node_input_detections(node) -> List[dai.ImgDetections]:
    return node.inputs[Node.INPUT_DETECTIONS_KEY].items


@pytest.mark.parametrize("platform", ["rvc2", "rvc4"])
def test_passthrough(
    node,
    node_input_detections,
    node_input_frames,
    platform,
    resize_width,
    resize_height,
):
    script = generate_script_content(platform, resize_width, resize_height)
    expected_frames = []
    for frame, detections in zip(node_input_frames, node_input_detections):
        for _ in detections.detections:
            expected_frames.append(frame)
    try:
        run_script(node, script)
    except Warning as w:
        assert w.args[0] == "pop from empty list"
        assert all([input.empty() for input in node.inputs.values()])
        assert get_output_frames(node) == expected_frames
        assert len(get_output_config(node)) == len(expected_frames)


@pytest.mark.parametrize(("platform", "labels"), [("rvc2", [1]), ("rvc4", [1, 2])])
def test_label_validation(
    node,
    node_input_detections,
    node_input_frames,
    platform,
    labels,
    resize_width,
    resize_height,
):
    expected_frames: List[Frame] = []
    for detections, frame in zip(node_input_detections, node_input_frames):
        for detection in detections.detections:
            if detection.label not in labels:
                continue
            expected_frames.append(frame)
    script = generate_script_content(
        platform, resize_width, resize_height, valid_labels=labels
    )
    try:
        run_script(node, script)
    except Warning:
        assert expected_frames == get_output_frames(node)


@pytest.mark.parametrize("resize", [(128, 128), (128, 256), (256, 256)])
def test_rvc2_output_size(node, resize):
    script = generate_script_content("rvc2", *resize)
    try:
        run_script(node, script)
    except Warning:
        output_cfg = get_output_config(node)
        for cfg in output_cfg:
            assert isinstance(cfg, dai.ImageManipConfig)
            assert cfg.getResizeWidth(), cfg.getResizeHeight() == resize


@pytest.mark.parametrize("resize", [(128, 128), (128, 256), (256, 256)])
def test_rvc4_output_size(node, resize):
    script = generate_script_content("rvc4", *resize)
    try:
        run_script(node, script)
    except Warning:
        output_cfg = get_output_config(node)
        for cfg in output_cfg:
            assert isinstance(cfg, ImageManipConfigV2)
            assert cfg.getOutputSize() == resize


@pytest.mark.parametrize("padding", [0, 0.1, 0.2, -0.1, -0.2])
def test_rvc2_crop(node, node_input_detections, padding, resize_width, resize_height):
    expected_rects: List[dai.ImageManipConfig.CropRect] = []
    for input_dets in node_input_detections:
        for detection in input_dets.detections:
            rect = dai.ImageManipConfig.CropRect()
            rect.xmin = max(detection.xmin - padding, 0)
            rect.xmax = min(detection.xmax + padding, 1)
            rect.ymin = max(detection.ymin - padding, 0)
            rect.ymax = min(detection.ymax + padding, 1)
            expected_rects.append(rect)
    script = generate_script_content(
        "rvc2", resize_width, resize_height, padding=padding
    )
    try:
        run_script(node, script)
    except Warning:
        output_cfg = get_output_config(node)
        for cfg, expected_rect in zip(output_cfg, expected_rects):
            assert isinstance(cfg, dai.ImageManipConfig)
            crop_rect = cfg.getCropConfig().cropRect
            assert (crop_rect.xmin, crop_rect.xmax, crop_rect.ymin, crop_rect.ymax) == (
                expected_rect.xmin,
                expected_rect.xmax,
                expected_rect.ymin,
                expected_rect.ymax,
            )


@pytest.mark.parametrize("padding", [0, 0.1, 0.2, -0.1, -0.2])
def test_rvc4_crop(node, node_input_detections, padding, resize_width, resize_height):
    ANGLE = 0
    expected_rects: List[dai.RotatedRect] = []
    for input_dets in node_input_detections:
        for detection in input_dets.detections:
            rect = dai.RotatedRect()
            rect.angle = ANGLE
            rect.center.x = (detection.xmin + detection.xmax) / 2
            rect.center.y = (detection.ymin + detection.ymax) / 2
            rect_padding = padding * 2
            rect.size.width = detection.xmax - detection.xmin + rect_padding
            rect.size.height = detection.ymax - detection.ymin + rect_padding
            expected_rects.append(rect)
    script = generate_script_content(
        "rvc4", resize_width, resize_height, padding=padding
    )
    try:
        run_script(node, script)
    except Warning:
        output_cfg = get_output_config(node)
        for cfg, expected_rect in zip(output_cfg, expected_rects):
            assert isinstance(cfg, ImageManipConfigV2)
            rotated_rect = cfg.getCropRotatedRect()
            assert rotated_rect
            assert rotated_rect.angle == expected_rect.angle
            assert (rotated_rect.size.width, rotated_rect.size.height) == pytest.approx(
                (expected_rect.size.width, expected_rect.size.height)
            )
            assert (rotated_rect.center.x, rotated_rect.center.y) == pytest.approx(
                (expected_rect.center.x, expected_rect.center.y)
            )


def run_script(node, script):
    exec(
        script,
        None,
        {
            "node": node,
            "ImageManipConfigV2": ImageManipConfigV2,
            "ImageManipConfig": dai.ImageManipConfig,
            "RotatedRect": dai.RotatedRect,
        },
    )


def get_output_frames(node: Node) -> List[Frame]:
    return node.outputs[Node.OUTPUT_FRAMES_KEY].items


def get_output_config(
    node: Node,
) -> Union[List[dai.ImageManipConfig], List[ImageManipConfigV2]]:
    return node.outputs[Node.OUTPUT_CONFIG_KEY].items
