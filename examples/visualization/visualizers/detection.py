import cv2
import depthai as dai
import numpy as np

from depthai_nodes.ml.messages import (
    Clusters,
    CornerDetections,
    ImgDetectionsExtended,
    Lines,
)

from .utils.colors import get_yolo_colors
from .utils.message_parsers import (
    parse_cluster_message,
    parse_corner_detection_message,
    parse_detection_message,
    parse_line_detection_message,
    parse_yolo_kpts_message,
)


def visualize_detections(
    frame: dai.ImgFrame, message: dai.ImgDetections, extraParams: dict
):
    """Visualizes the detections on the frame.

    Also, checks if there are any keypoints available to visualize.
    """
    labels = extraParams.get("classes", None)
    detections = parse_detection_message(message)
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

        try:
            keypoints = detection.keypoints
            for kp in keypoints:
                cv2.circle(
                    frame,
                    (int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])),
                    5,
                    (0, 0, 255),
                    -1,
                )
        except Exception:
            pass

        cv2.putText(
            frame,
            f"{detection.confidence * 100:.2f}%",
            (int(xmin) + 10, int(ymin) + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        if labels is not None:
            cv2.putText(
                frame,
                labels[detection.label],
                (int(xmin) + 10, int(ymin) + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_line_detections(frame: dai.ImgFrame, message: Lines, extraParams: dict):
    """Visualizes the lines on the frame."""
    lines = parse_line_detection_message(message)
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
    frame: dai.ImgFrame, message: ImgDetectionsExtended, extraParams: dict
):
    """Visualizes the YOLO pose detections or instance segmentation on the frame."""
    detections = parse_yolo_kpts_message(message)

    colors = get_yolo_colors()
    classes = extraParams.get("classes", None)
    if classes is None:
        raise ValueError("Classes are required for visualization.")
    task = extraParams.get("n_keypoints", None)
    if task is None:
        task = "segmentation"
    else:
        task = "keypoints"

    overlay = np.zeros_like(frame)

    for detection in detections:
        xmin, ymin, xmax, ymax = (
            detection.xmin,
            detection.ymin,
            detection.xmax,
            detection.ymax,
        )
        cv2.rectangle(
            frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2
        )
        cv2.putText(
            frame,
            f"{detection.confidence * 100:.2f}%",
            (int(xmin) + 10, int(ymin) + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        cv2.putText(
            frame,
            f"{classes[detection.label]}",
            (int(xmin) + 10, int(ymin) + 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )

        if task == "keypoints":
            keypoints = detection.keypoints
            for keypoint in keypoints:
                x, y, visibility = keypoint[0], keypoint[1], keypoint[2]
                if visibility > 0.8:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        else:
            mask = detection.mask
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask = detection.mask
            mask = cv2.resize(mask, (512, 288))
            label = detection.label
            overlay[mask > 0] = colors[(label % len(colors))]

    if task == "segmentation":
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.5, 0, None)
    cv2.imshow("YOLO Pose Estimation", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_lane_detections(
    frame: dai.ImgFrame, message: Clusters, extraParams: dict
):
    """Visualizes the lines on the frame."""
    clusters = parse_cluster_message(message)

    for cluster in clusters:
        for point in cluster.points:
            x = int(point.x)
            y = int(point.y)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # draw lines between points
    for cluster in clusters:
        for i in range(len(cluster.points) - 1):
            cv2.line(
                frame,
                (int(cluster.points[i].x), int(cluster.points[i].y)),
                (int(cluster.points[i + 1].x), int(cluster.points[i + 1].y)),
                (0, 255, 0),
                2,
            )

    cv2.imshow("Lane Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_text_detection(
    frame: dai.ImgFrame, message: CornerDetections, extraParams: dict
):
    detections = parse_corner_detection_message(message)
    for detection in detections:
        for i in range(len(detection.keypoints)):
            cv2.circle(
                frame,
                (int(detection.keypoints[i].x), int(detection.keypoints[i].y)),
                3,
                (0, 255, 0),
                -1,
            )
            if i == len(detection.keypoints) - 1:
                cv2.line(
                    frame,
                    (int(detection.keypoints[i].x), int(detection.keypoints[i].y)),
                    (int(detection.keypoints[0].x), int(detection.keypoints[0].y)),
                    (0, 255, 0),
                    2,
                )
                break
            cv2.line(
                frame,
                (int(detection.keypoints[i].x), int(detection.keypoints[i].y)),
                (int(detection.keypoints[i + 1].x), int(detection.keypoints[i + 1].y)),
                (0, 255, 0),
                2,
            )

    cv2.imshow("Text Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
