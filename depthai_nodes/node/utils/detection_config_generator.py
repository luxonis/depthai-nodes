import textwrap


def generate_script_content(
    resize_width: int,
    resize_height: int,
    resize_mode: str = "STRETCH",
    padding: float = 0.0,
) -> str:
    """Generates the script content for the dai.Script node.

    It crops and resizes the input image based on the detected object, with optional
    padding and label filtering. If a zero-area detection is encountered, an error
    message is issued.

    @param resize_width: Target width for the resized image
    @type resize_width: int
    @param resize_height: Target height for the resized image
    @type resize_height: int
    @param resize_mode: Resize mode for the image. Supported values: "CENTER_CROP",
        "LETTERBOX", "NONE", "STRETCH". Default: "STRETCH".
        "STRETCH" - stretches the image so that the corners of the region are now in the corners of the output image.
        "CENTER_CROP" - resizes + crops the image to keep aspect ratio and fill the final size.
        "LETTERBOX" - resizes + pads the image to the final size to keep aspect ratio.
        "NONE" - does not scale and pads top, bottom, left and right to fill the final image.
    @type resize_mode: str
    @param padding: Additional padding around the detection in normalized coordinates
        (0-1)
    @type padding: float
    @param valid_labels: List of valid label indices to filter detections. If None, all
        detections are processed
    @type valid_labels: Optional[List[int]]
    @return: Generated script content as a string
    @rtype: str
    """

    if resize_mode not in ["CENTER_CROP", "LETTERBOX", "NONE", "STRETCH"]:
        raise ValueError("Unsupported resize mode")

    script_content = f"""\
try:
    while True:
        frame = node.inputs['preview'].get()
        dets = node.inputs['det_in'].get()

        for i, det in enumerate(dets.detections):
            cfg = ImageManipConfig()
            rect = RotatedRect()

            rect.center.x = (det.xmin + det.xmax) / 2
            rect.center.y = (det.ymin + det.ymax) / 2
            rect.size.width = (det.xmax - det.xmin) + {padding} * 2
            rect.size.height = (det.ymax - det.ymin) + {padding} * 2
            rect.angle = 0

            # Detect zero-area detections and issue an error
            if rect.size.width <= 0.0 or rect.size.height <= 0.0:
                raise ValueError(
                    f"Got zero-area detection (w={{rect.size.width}}, h={{rect.size.height}}). "
                    f"Consider using ImgDetectionsFilter with min_area > 0 "
                    f"to exclude such detections before cropping."
                )

            cfg.addCropRotatedRect(rect, True)
            cfg.setOutputSize({resize_width}, {resize_height}, ImageManipConfig.ResizeMode.{resize_mode})
            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)

except Exception as e:
    node.warn(str(e))
"""

    return textwrap.dedent(script_content).lstrip()
