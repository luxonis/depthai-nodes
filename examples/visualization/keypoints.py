import cv2
import depthai as dai

from depthai_nodes.ml.messages import Keypoints

from .messages import parse_keypoints_message


def visualize_keypoints(frame: dai.ImgFrame, message: Keypoints, extraParams: dict):
    """Visualizes the keypoints on the frame."""
    keypoints = parse_keypoints_message(message)

    for kp in keypoints:
        x = int(kp.x * frame.shape[1])
        y = int(kp.y * frame.shape[0])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Keypoints", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
