import cv2
import depthai as dai

from .messages import parse_image_message


def visualize_image(frame: dai.ImgFrame, message: dai.ImgFrame, extraParams: dict):
    """Visualizes the image on the frame."""
    image = parse_image_message(message)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
