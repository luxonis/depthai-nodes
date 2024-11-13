import cv2
import numpy as np

from depthai_nodes.ml.messages import Classifications


def visualize_classification(
    frame: np.ndarray, message: Classifications, extraParams: dict
):
    """Visualizes the classification on the frame."""
    classes = message.classes[:2]
    scores = message.scores[:2]
    if frame.shape[0] < 128:
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
    for i, (cls, score) in enumerate(zip(classes, scores)):
        cv2.putText(
            frame,
            f"{cls}: {score:.2f}",
            (10, 20 + 20 * i),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )

    cv2.imshow("Classification", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_text_recognition(
    frame: np.ndarray, message: Classifications, extraParams: dict
):
    """Visualizes the text recognition on the frame."""

    if frame.shape[0] < 128:
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

    classes = message.classes
    text = "".join(classes)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

    cv2.imshow("Text recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
