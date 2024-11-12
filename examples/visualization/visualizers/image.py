import cv2
import depthai as dai


def visualize_image(frame: dai.ImgFrame, message: dai.ImgFrame, extraParams: dict):
    """Visualizes the image on the frame."""
    image = message.getFrame()
    cv2.imshow("Image", image)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
