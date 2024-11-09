import cv2
import depthai as dai
import numpy as np

from depthai_nodes.ml.messages import (
    Clusters,
    ImgDetectionsExtended,
    Lines,
)

from .utils.message_parsers import (
    parse_cluster_message,
    parse_detection_message,
    parse_line_detection_message,
    parse_yolo_kpts_message,
)


def visualize_detections(
    frame: np.ndarray, message: ImgDetectionsExtended, extraParams: dict
):
    """Visualizes the detections on the frame. Detections are given in xywh format
    (ImgDetectionsExtended).

    Also, checks if there are any keypoints available to visualize.
    """
    labels = extraParams.get("classes", None)
    detections = parse_detection_message(message)
    for detection in detections:
        x_center, y_center, width, height = (
            detection.x_center,
            detection.y_center,
            detection.width,
            detection.height,
        )
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2

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

        if detection.angle == 0:
            cv2.rectangle(
                frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2
            )
        else:
            # Rotated rectangle
            x_center = int(x_center * frame.shape[1])
            y_center = int(y_center * frame.shape[0])
            width = int(width * frame.shape[1])
            height = int(height * frame.shape[0])
            box = cv2.boxPoints(
                ((x_center, y_center), (width, height), detection.angle)
            )
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

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


def visualize_detections_xyxy(
    frame: dai.ImgFrame, message: dai.ImgDetections, extraParams: dict
):
    """Visualize the detections on the frame.

    The detections are in xyxy format (dai.ImgDetections).
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

    classes = extraParams.get("classes", None)
    if classes is None:
        raise ValueError("Classes are required for visualization.")
    task = extraParams.get("n_keypoints", None)
    if task is None:
        task = "segmentation"
    else:
        task = "keypoints"

    for detection in detections:
        x_center, y_center, width, height = (
            detection.x_center,
            detection.y_center,
            detection.width,
            detection.height,
        )
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2
        cv2.rectangle(
            frame,
            (int(xmin * frame.shape[1]), int(ymin * frame.shape[0])),
            (int(xmax * frame.shape[1]), int(ymax * frame.shape[0])),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{detection.confidence * 100:.2f}%",
            (int(xmin * frame.shape[1]) + 10, int(ymin * frame.shape[0]) + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        cv2.putText(
            frame,
            f"{classes[detection.label]}",
            (int(xmin * frame.shape[1]) + 10, int(ymin * frame.shape[0]) + 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )

        if task == "keypoints":
            keypoints = detection.keypoints
            for keypoint in keypoints:
                x, y, visibility = (
                    keypoint.x * frame.shape[1],
                    keypoint.y * frame.shape[0],
                    keypoint.confidence,
                )
                if visibility > 0.8:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    if task == "segmentation":
        mask = message.masks
        mask = mask > -0.5
        frame[mask] = frame[mask] * 0.5 + np.array((0, 255, 0)) * 0.5
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
