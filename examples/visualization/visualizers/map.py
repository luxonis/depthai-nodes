import cv2
import depthai as dai
import numpy as np

from depthai_nodes.ml.messages import Map2D

from .utils.message_parsers import parse_map_message


def visualize_map(frame: np.ndarray, message: Map2D, extraParams: dict):
    """Visualizes the map on the frame."""

    map = parse_map_message(message)

    # make color representation of the map
    map_normalized = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX)
    map_normalized = map_normalized.astype(np.uint8)
    colored_map = cv2.applyColorMap(map_normalized, cv2.COLORMAP_INFERNO)
    frame_height, frame_width, _ = frame.shape
    colored_map = cv2.resize(
        colored_map, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR
    )

    alpha = 0.6
    overlay = cv2.addWeighted(colored_map, alpha, frame, 1 - alpha, 0)

    cv2.imshow("Map Overlay", overlay)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
