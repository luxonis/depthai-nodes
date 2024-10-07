import cv2
import depthai as dai
import numpy as np

from depthai_nodes.ml.messages import Classifications, CompositeMessage

from .utils.message_parsers import (
    parse_classification_message,
    parse_multi_classification_message,
    parser_age_gender_message,
)


def visualize_classification(
    frame: dai.ImgFrame, message: Classifications, extraParams: dict
):
    """Visualizes the classification on the frame."""
    classes, scores = parse_classification_message(message)
    classes = classes[:2]
    scores = scores[:2]
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


def visualize_multi_classification(
    frame: dai.ImgFrame, message: Classifications, extraParams: dict
):
    """Visualizes the multi classification on the frame."""
    parsed_message = parse_multi_classification_message(message)
    if frame.shape[0] < 128:
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

    for i, atribute in enumerate(parsed_message.keys()):
        classes = parsed_message[atribute]["classes"]
        scores = parsed_message[atribute]["scores"]
        predicted_class = classes[np.argmax(scores)]
        cv2.putText(
            frame,
            f"{atribute}: {predicted_class}",
            (10, 20 + 20 * i),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.3,
            255,
        )

    cv2.imshow("Multi Classification", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_age_gender(
    frame: dai.ImgFrame, message: CompositeMessage, extraParams: dict
):
    """Visualizes the age and predicted gender on the frame."""
    if frame.shape[0] < 128:
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
    age, gender_classes, gender_scores = parser_age_gender_message(message)
    cv2.putText(frame, f"Age: {age}", (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    for i, (cls, score) in enumerate(zip(gender_classes, gender_scores)):
        cv2.putText(
            frame,
            f"{cls}: {score:.2f}",
            (10, 40 + 20 * i),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )

    cv2.imshow("Age-gender", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False


def visualize_text_recognition(
    frame: dai.ImgFrame, message: CompositeMessage, extraParams: dict
):
    """Visualizes the text recognition on the frame."""

    if frame.shape[0] < 128:
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

    classes, scores = parse_classification_message(message)
    text = "".join(classes)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

    cv2.imshow("Text recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
