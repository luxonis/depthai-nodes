import numpy as np

from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    compute_yolo_detections,
)


def test_compute_yolo_detections_uses_anchors_per_head_for_class_count(
    monkeypatch,
):
    monkeypatch.setattr(
        "depthai_nodes.node.parsers.utils.yolo.decode_yolo_output",
        lambda *args, **kwargs: np.zeros((0, 6), dtype=np.float32),
    )

    outputs_values = [
        np.zeros((1, 255, 26, 26), dtype=np.float32),
        np.zeros((1, 255, 13, 13), dtype=np.float32),
    ]
    anchors = [
        [[10.0, 14.0], [23.0, 27.0], [37.0, 58.0]],
        [[81.0, 82.0], [135.0, 169.0], [344.0, 319.0]],
    ]

    payload = compute_yolo_detections(
        subtype=YOLOSubtype.V3T,
        layer_names=["output1_yolo", "output2_yolo"],
        outputs_values=outputs_values,
        strides=[16, 32],
        conf_threshold=0.25,
        n_classes=80,
        iou_threshold=0.45,
        max_det=300,
        anchors=anchors,
        n_keypoints=17,
        label_names=[f"class_{idx}" for idx in range(80)],
        keypoint_label_names=None,
        keypoint_edges=None,
        input_shape=(416, 416),
    )

    assert payload["scores"].size == 0
    assert payload["labels"].size == 0
