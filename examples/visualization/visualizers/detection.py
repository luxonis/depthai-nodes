from typing import List

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.ml.messages import (
    Clusters,
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Lines,
)


def draw_extended_image_detections(
    frame: np.ndarray, detections: List[ImgDetectionExtended], labels=None
):
    """Draws (rotated) bounding boxes on the given frame."""

    for detection in detections:
        rect = detection.rotated_rect
        points = rect.getPoints()

        bbox = np.array([[point.x, point.y] for point in points])
        if np.any(bbox < 1):
            bbox[:, 0] = bbox[:, 0] * frame.shape[1]
            bbox[:, 1] = bbox[:, 1] * frame.shape[0]
        bbox = bbox.astype(int)
        cv2.polylines(frame, [bbox], isClosed=True, color=(255, 0, 0), thickness=2)

        try:
            keypoints = detection.keypoints
            for kp in keypoints:
                cv2.circle(
                    frame,
                    (int(kp.x * frame.shape[1]), int(kp.y * frame.shape[0])),
                    5,
                    (0, 0, 255),
                    -1,
                )
        except Exception:
            pass

        outer_points = rect.getOuterRect()
        xmin = int(outer_points[0] * frame.shape[1])
        ymin = int(outer_points[1] * frame.shape[0])
        cv2.putText(
            frame,
            f"{detection.confidence * 100:.2f}%",
            (xmin + 10, ymin + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            (255, 0, 0),
        )
        if labels is not None:
            cv2.putText(
                frame,
                labels[detection.label],
                (xmin + 10, ymin + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (255, 0, 0),
            )

    return frame


def visualize_detections(
    frame: np.ndarray, message: ImgDetectionsExtended, extraParams: dict
):
    """Visualizes the detections on the frame. Detections are given in x_center, width,
    height, angle format (ImgDetectionsExtended).

    Also, checks if there are any keypoints available to visualize.
    """
    labels = extraParams.get("classes", None)
    detections = message.detections
    frame = draw_extended_image_detections(frame, detections, labels)

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_detections_xyxy(
    frame: np.ndarray, message: dai.ImgDetections, extraParams: dict
):
    """Visualize the detections on the frame.

    The detections are in xyxy format (dai.ImgDetections).
    """

    labels = extraParams.get("classes", None)
    detections = message.detections
    for detection in detections:
        xmin, ymin, xmax, ymax = (
            detection.xmin,
            detection.ymin,
            detection.xmax,
            detection.ymax,
        )
        if xmin > 1 or ymin > 1 or xmax > 1 or ymax > 1:
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
        else:
            xmin = int(xmin * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            xmax = int(xmax * frame.shape[1])
            ymax = int(ymax * frame.shape[0])
        cv2.rectangle(
            frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2
        )

        cv2.putText(
            frame,
            f"{detection.confidence * 100:.2f}%",
            (int(xmin) + 10, int(ymin) + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            (255, 0, 0),
        )
        if labels is not None:
            cv2.putText(
                frame,
                labels[detection.label],
                (int(xmin) + 10, int(ymin) + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (255, 0, 0),
            )

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_line_detections(frame: np.ndarray, message: Lines, extraParams: dict):
    """Visualizes the lines on the frame."""
    lines = message.lines
    h, w = frame.shape[:2]
    for line in lines:
        x1 = line.start_point.x * w
        y1 = line.start_point.y * h
        x2 = line.end_point.x * w
        y2 = line.end_point.y * h
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3, 16)

    cv2.putText(
        frame,
        f"Number of lines: {len(lines)}",
        (2, frame.shape[0] - 4),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (255, 0, 0),
    )
    cv2.imshow("Lines", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_yolo_extended(
    frame: np.ndarray, message: ImgDetectionsExtended, extraParams: dict
):
    """Visualizes the YOLO pose detections or instance segmentation on the frame."""

    detections = message.detections
    classes = extraParams.get("classes", None)

    if classes is None:
        raise ValueError("Classes are required for visualization.")
    task = extraParams.get("n_keypoints", None)
    if task is None:
        task = "segmentation"

    frame = draw_extended_image_detections(frame, detections, classes)

    if task == "segmentation":
        mask = message.masks
        mask = mask > -0.5
        frame[mask] = frame[mask] * 0.5 + np.array((0, 255, 0)) * 0.5

    cv2.imshow("YOLO Pose Estimation", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_lane_detections(frame: np.ndarray, message: Clusters, extraParams: dict):
    """Visualizes the lines on the frame."""
    clusters = message.clusters
    h, w, _ = frame.shape
    for cluster in clusters:
        for point in cluster.points:
            x = int(point.x * w)
            y = int(point.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # draw lines between points
    for cluster in clusters:
        for i in range(len(cluster.points) - 1):
            cv2.line(
                frame,
                (int(cluster.points[i].x * w), int(cluster.points[i].y * h)),
                (int(cluster.points[i + 1].x * w), int(cluster.points[i + 1].y * h)),
                (0, 255, 0),
                2,
            )

    cv2.imshow("Lane Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
