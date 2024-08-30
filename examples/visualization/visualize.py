import depthai as dai

from .mapping import parser_mapping


def visualize(
    frame: dai.ImgFrame, message: dai.Buffer, parser_name: str, extraParams: dict
):
    """Calls the appropriate visualizer based on the parser name and returns True if the
    pipeline should be stopped."""
    visualizer = parser_mapping[parser_name]
    return visualizer(frame, message, extraParams)
