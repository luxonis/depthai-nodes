import cv2
import depthai as dai
import numpy as np


def visualize_image(frame: np.ndarray, message: dai.ImgFrame, extraParams: dict):
    """Visualizes the image on the frame."""
    image = message.getFrame()
    cv2.imshow("Image", image)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
