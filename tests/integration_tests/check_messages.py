import pickle
from typing import Any, Dict, List

import depthai as dai
import numpy as np
from utils import extract_main_slug

from depthai_nodes.ml.messages import (
    Classifications,
    Clusters,
    ImgDetectionsExtended,
    Keypoints,
    Lines,
    Map2D,
    SegmentationMask,
)


def load_expected_output(model: str, parser: str) -> Dict[str, Any]:
    model = extract_main_slug(model)
    with open(f"nn_datas/{parser}/{model}_output.pkl", "rb") as f:
        return pickle.load(f)


def check_classification_msg(message: Classifications, expected_output: Dict[str, Any]):
    print(
        f"Expected top class: {expected_output['class']}, predicted top class: {message.top_class}"
    )
    print(
        f"Expected top score: {expected_output['score']}, predicted top score: {message.top_score}"
    )
    assert message.top_class == expected_output["class"]
    np.testing.assert_allclose(message.top_score, expected_output["score"], rtol=1e-2)


def check_classification_sequence_msg(
    message: Classifications, expected_output: Dict[str, Any]
):
    print(
        f"Expected top class: {expected_output['class']}, predicted top class: {message.classes}"
    )
    assert len(message.classes) == len(
        expected_output["class"]
    ), "The number of classes is different."
    for i in range(len(message.classes)):
        assert (
            message.classes[i] == expected_output["class"][i]
        ), f"Class {i} is different."


def check_embeddings_msg(message: dai.NNData, expected_output: Dict[str, Any]):
    outputs = message.getAllLayerNames()
    assert len(outputs) == 1, "The number of outputs must be 1."
    embeddings = message.getTensor(outputs[0])
    expected_embeddings = expected_output["embeddings"]
    print(
        f"Expected embeddings shape: {expected_embeddings.shape}, predicted embeddings shape: {embeddings.shape}"
    )
    assert (
        embeddings.shape == expected_embeddings.shape
    ), "The shape of the embeddings is different."
    np.testing.assert_allclose(embeddings, expected_embeddings, rtol=1e-2)


def check_fastsam_msg(
    message: SegmentationMask, expected_output: Dict[str, Any], threshold: float = 0.9
):
    mask = message.mask

    expected_mask = expected_output["mask"]
    print(
        f"Expected mask shape: {expected_mask.shape}, predicted mask shape: {mask.shape}"
    )
    assert (
        mask.shape == expected_mask.shape
    ), f"The shape of the mask is different. Expects {expected_mask.shape}, got {mask.shape}"
    assert (
        mask.dtype == expected_mask.dtype
    ), f"The dtype of the mask is different. Expects {expected_mask.dtype}, got {mask.dtype}"

    correct = (mask == expected_mask).sum()
    total = expected_mask.size
    acc = correct / total
    print(f"Accuracy: {acc}")
    assert (
        acc > threshold
    ), f"The accuracy {acc} is lower than the threshold {threshold}"


def check_keypoints_msg(
    message: Keypoints,
    expected_output: Dict[str, Any],
    distance_threshold: float = 0.001,
):
    keypoints = [[kp.x, kp.y] for kp in message.keypoints]
    keypoints = np.array(keypoints)
    expected_keypoints = np.array(expected_output["keypoints"])
    if expected_keypoints.shape[1] == 3:
        expected_keypoints = expected_keypoints[:, :2]  # we dont need the confidences
    print(
        f"Expected keypoints shape: {expected_keypoints.shape}, predicted keypoints shape: {keypoints.shape}"
    )
    assert (
        keypoints.shape == expected_keypoints.shape
    ), f"The shape of the keypoints is different. Expects {expected_keypoints.shape}, got {keypoints.shape}"

    for gt_kpt in expected_keypoints:
        min_dist = float("inf")
        for pred_kpt in keypoints:
            dist = (gt_kpt[0] - pred_kpt[0]) ** 2 + (gt_kpt[1] - pred_kpt[1]) ** 2
            if dist < min_dist:
                min_dist = dist
        assert (
            min_dist < distance_threshold
        ), f"Expected min distance < {distance_threshold}, got {min_dist}"


def check_image_msg(message: dai.ImgFrame, expected_output: Dict[str, Any]):
    image = message.getCvFrame()
    expected_image = expected_output["output"]
    print(
        f"Expected image shape: {expected_image.shape}, predicted image shape: {image.shape}"
    )
    assert (
        image.shape == expected_image.shape
    ), f"The shape of the image is different. Expects {expected_image.shape}, got {image.shape}"
    assert np.allclose(image, expected_image, atol=1), "The image is different."


def check_cluster_msg(message: Clusters, expected_output: Dict[str, Any]):
    clusters = message.clusters
    expected_clusters = expected_output["clusters"]
    clusters_list = []
    for cluster in clusters:
        c = []
        for point in cluster.points:
            c.append([point.x, point.y])
        clusters_list.append(c)

    # check if all points are in the expected clusters
    for i, cluster in enumerate(expected_clusters):
        for gt_point in cluster:
            found = False
            for pred_point in clusters_list[i]:
                if np.allclose(gt_point, pred_point, atol=0.001):
                    found = True
                    break
            assert found, f"Expected point {gt_point} not found in the cluster."


def check_map_msg(message: Map2D, expected_output: Dict[str, Any]):
    map_tensor = message.map
    expected_map = expected_output["map"]
    print(
        f"Expected map shape: {expected_map.shape}, predicted map shape: {map_tensor.shape}"
    )
    assert (
        map_tensor.shape == expected_map.shape
    ), f"The shape of the map is different. Expects {expected_map.shape}, got {map_tensor.shape}"
    assert np.allclose(map_tensor, expected_map, atol=0.001), "The map is different."


def check_detection_msg(
    message: ImgDetectionsExtended, expected_output: Dict[str, Any]
):
    predicted_detections = []
    expected_detections: List[Dict[str, Any]] = expected_output["detections"]

    for detection in message.detections:
        d = {
            "confidence": detection.confidence,
            "label": detection.label,
            "x_center": detection.rotated_rect.center.x,
            "y_center": detection.rotated_rect.center.y,
            "width": detection.rotated_rect.size.width,
            "height": detection.rotated_rect.size.height,
            "angle": detection.rotated_rect.angle,
            "keypoints": [[kp.x, kp.y, kp.confidence] for kp in detection.keypoints],
            "mask": message.masks,
        }
        predicted_detections.append(d)

    assert (
        len(predicted_detections) == len(expected_detections)
    ), f"The number of detections is different. Got {len(predicted_detections)}, expected {len(expected_detections)}"
    print(
        f"Expected number of detections: {len(expected_detections)}, predicted number of detections: {len(predicted_detections)}"
    )

    for i, expected_detection in enumerate(expected_detections):
        predicted_detection = predicted_detections[i]
        assert np.allclose(
            expected_detection["confidence"],
            predicted_detection["confidence"],
            atol=0.01,
        ), f"Expected confidence {expected_detection['confidence']}, got {predicted_detection['confidence']}."
        assert (
            expected_detection["label"] == predicted_detection["label"]
        ), f"Expected label {expected_detection['label']}, got {predicted_detection['label']}."
        assert np.allclose(
            expected_detection["x_center"], predicted_detection["x_center"], atol=0.01
        ), f"Expected x_center {expected_detection['x_center']}, got {predicted_detection['x_center']}."
        assert np.allclose(
            expected_detection["y_center"], predicted_detection["y_center"], atol=0.01
        ), f"Expected y_center {expected_detection['y_center']}, got {predicted_detection['y_center']}."
        assert np.allclose(
            expected_detection["width"], predicted_detection["width"], atol=0.01
        ), f"Expected width {expected_detection['width']}, got {predicted_detection['width']}."
        assert np.allclose(
            expected_detection["height"], predicted_detection["height"], atol=0.01
        ), f"Expected height {expected_detection['height']}, got {predicted_detection['height']}."
        assert np.allclose(
            expected_detection["angle"], predicted_detection["angle"], atol=0.01
        ), f"Expected angle {expected_detection['angle']}, got {predicted_detection['angle']}."

        if "keypoints" in expected_detection.keys():
            assert np.allclose(
                expected_detection["keypoints"],
                predicted_detection["keypoints"],
                atol=0.01,
            ), f"Expected keypoints {expected_detection['keypoints']}, got {predicted_detection['keypoints']}."

        if "mask" in expected_detection.keys():
            print(
                f"Expected mask shape: {expected_detection['mask'].shape}, predicted mask shape: {predicted_detection['mask'].shape}"
            )
            assert (
                expected_detection["mask"].shape == predicted_detection["mask"].shape
            ), f"The shape of the mask is different. Expects {expected_detection['mask'].shape}, got {predicted_detection['mask'].shape}"
            assert np.allclose(
                expected_detection["mask"], predicted_detection["mask"], atol=0.01
            ), f"Expected mask {expected_detection['mask']}, got {predicted_detection['mask']}."


def check_line_msg(message: Lines, expected_output: Dict[str, Any]):
    """
    Expected output format:
    {
        "model": "luxonis/m-mlsd-tiny:512x512",
        "parser": "MLSDParser",
        "lines": [
            {
                "confidence": 0.9,
                "start_point": [0.1, 0.2],
                "end_point": [0.3, 0.4]
            },
            ...
        ]
    }
    """
    expected_lines: List[Dict[str, Any]] = expected_output["lines"]
    predicted_lines = []
    for line in message.lines:
        line_dict = {
            "confidence": line.confidence,
            "start_point": [line.start_point.x, line.start_point.y],
            "end_point": [line.end_point.x, line.end_point.y],
        }
        predicted_lines.append(line_dict)

    print(
        f"Expected number of lines: {len(expected_lines)}, predicted number of lines: {len(predicted_lines)}"
    )
    assert (
        len(predicted_lines) == len(expected_lines)
    ), f"The number of lines is different. Got {len(predicted_lines)}, expected {len(expected_lines)}"

    for i, expected_line in enumerate(expected_lines):
        predicted_line = predicted_lines[i]
        assert np.allclose(
            expected_line["confidence"], predicted_line["confidence"], atol=0.01
        ), f"Expected confidence {expected_line['confidence']}, got {predicted_line['confidence']}."
        assert np.allclose(
            expected_line["start_point"], predicted_line["start_point"], atol=0.01
        ), f"Expected start_point {expected_line['start_point']}, got {predicted_line['start_point']}."
        assert np.allclose(
            expected_line["end_point"], predicted_line["end_point"], atol=0.01
        ), f"Expected end_point {expected_line['end_point']}, got {predicted_line['end_point']}."


def test_output(message, model_slug, parser_name):
    expected_output = load_expected_output(model_slug, parser_name)

    if expected_output["parser"] == "ClassificationParser":
        check_classification_msg(message, expected_output)
    elif expected_output["parser"] == "ClassificationSequenceParser":
        check_classification_sequence_msg(message, expected_output)
    elif expected_output["parser"] == "EmbeddingsParser":
        check_embeddings_msg(message, expected_output)
    elif expected_output["parser"] == "FastSAMParser":
        check_fastsam_msg(message, expected_output)
    elif expected_output["parser"] == "HRNetParser":
        check_keypoints_msg(message, expected_output)
    elif expected_output["parser"] == "KeypointParser":
        check_keypoints_msg(message, expected_output)
    elif expected_output["parser"] == "ImageOutputParser":
        check_image_msg(message, expected_output)
    elif expected_output["parser"] == "LaneDetectionParser":
        check_cluster_msg(message, expected_output)
    elif expected_output["parser"] == "MapOutputParser":
        check_map_msg(message, expected_output)
    elif expected_output["parser"] == "MPPalmDetectionParser":
        check_detection_msg(message, expected_output)
    elif expected_output["parser"] == "MLSDParser":
        check_line_msg(message, expected_output)
    elif expected_output["parser"] == "PPTextDetectionParser":
        check_detection_msg(message, expected_output)
    elif expected_output["parser"] == "SCRFDParser":
        check_detection_msg(message, expected_output)
    elif expected_output["parser"] == "SegmentationParser":
        check_fastsam_msg(message, expected_output)
    elif expected_output["parser"] == "SuperAnimalParser":
        check_keypoints_msg(message, expected_output)
    elif expected_output["parser"] == "YuNetParser":
        check_detection_msg(message, expected_output)
    elif expected_output["parser"] == "YOLOExtendedParser":
        check_detection_msg(message, expected_output)
