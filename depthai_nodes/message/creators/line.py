import depthai as dai
import numpy as np

from depthai_nodes import Line, Lines


def create_line_detection_message(lines: np.ndarray, scores: np.ndarray):
    """Create a DepthAI message for a line detection.

    @param lines: Detected lines of shape (N,4) meaning [...,[x_start, y_start, x_end, y_end],...].
    @type lines: np.ndarray
    @param scores: Confidence scores of detected lines of shape (N,).
    @type scores: np.ndarray

    @return: Message containing the lines and confidence scores of detected lines.
    @rtype: Lines

    @raise ValueError: If the lines are not a numpy array.
    @raise ValueError: If the lines are not of shape (N,4).
    @raise ValueError: If the lines 2nd dimension is not of size E{4}.
    @raise ValueError: If the scores are not a numpy array.
    @raise ValueError: If the scores are not of shape (N,).
    @raise ValueError: If the scores do not have the same length as lines.
    """

    # checks for lines
    if not isinstance(lines, np.ndarray):
        raise ValueError(f"Lines should be numpy array, got {type(lines)}.")
    if len(lines) != 0:
        if len(lines.shape) != 2:
            raise ValueError(
                f"Lines should be of shape (N,4) meaning [...,[x_start, y_start, x_end, y_end],...], got {lines.shape}."
            )
        if lines.shape[1] != 4:
            raise ValueError(
                f"Lines 2nd dimension should be of size 4 e.g. [x_start, y_start, x_end, y_end] got {lines.shape[1]}."
            )

    # checks for scores
    if not isinstance(scores, np.ndarray):
        raise ValueError(f"Scores should be numpy array, got {type(scores)}.")

    if len(scores) != 0:
        if len(scores.shape) != 1:
            raise ValueError(
                f"Scores should be of shape (N,) meaning, got {scores.shape}."
            )

        for score in scores:
            if not isinstance(score, (float, np.floating)):
                raise ValueError(f"Scores should be of type float, got {type(score)}.")

    if scores.shape[0] != lines.shape[0]:
        raise ValueError(
            f"Scores should have same length as lines, got {scores.shape[0]} and {lines.shape[0]}."
        )

    line_detections = []
    for i, line in enumerate(lines):
        line_detection = Line()
        line_detection.start_point = dai.Point2f(line[0], line[1])
        line_detection.end_point = dai.Point2f(line[2], line[3])
        line_detection.confidence = float(scores[i])
        line_detections.append(line_detection)

    lines_msg = Lines()
    lines_msg.lines = line_detections
    return lines_msg
