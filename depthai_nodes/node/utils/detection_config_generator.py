import textwrap
from typing import List, Optional


def generate_script_content(
    resize_width: int,
    resize_height: int,
    resize_mode: str = "STRETCH",
    padding: float = 0.0,
    valid_labels: Optional[List[int]] = None,
) -> str:
    """The function generates the script content for the dai.Script node.

    It is used to crop and resize the input image based on the detected object. It can
    also work with padding around the detection bounding box and filter detections by
    labels.

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

    cfg_core = f"""\
cfg = ImageManipConfig()
rect = RotatedRect()

# frame size (for epsilon). Works on device and in tests.
try:
    fw = float(frame.getWidth()); fh = float(frame.getHeight())
except Exception:
    try:
        fw = float(getattr(frame, "width")); fh = float(getattr(frame, "height"))
    except Exception:
        fw = float({resize_width}); fh = float({resize_height})
if fw <= 0: fw = float({resize_width})
if fh <= 0: fh = float({resize_height})

# read & order bbox
xmin = det.xmin; ymin = det.ymin; xmax = det.xmax; ymax = det.ymax
if xmin > xmax: xmin, xmax = xmax, xmin
if ymin > ymax: ymin, ymax = ymax, ymin

# legacy center + padding-as-size (no clamping)
cx = 0.5 * (xmin + xmax)
cy = 0.5 * (ymin + ymax)
w  = (xmax - xmin) + 2*({padding})
h  = (ymax - ymin) + 2*({padding})

# only rescue true zero-area (avoid ImageManip 'Colinear points')
if w == 0.0 or h == 0.0:
    safety_px = 2.0
    eps = max(safety_px/fw, safety_px/fh)
    if w == 0.0: w = eps
    if h == 0.0: h = eps

rect.center.x = cx
rect.center.y = cy
rect.size.width  = w
rect.size.height = h
rect.angle = 0

cfg.addCropRotatedRect(rect, True)
cfg.setOutputSize({resize_width}, {resize_height}, ImageManipConfig.ResizeMode.{resize_mode})
"""

    cfg_content = textwrap.indent(cfg_core, " " * 12)
    if valid_labels is not None:
        validate_label = f"if det.label not in {valid_labels}: continue\n"
        indented_validate_label = textwrap.indent(validate_label, " " * 12)
    else:
        indented_validate_label = ""

    outer = f"""\
try:
    while True:
        frame = node.inputs['preview'].get()
        dets  = node.inputs['det_in'].get()
        for det in dets.detections:
{indented_validate_label}
{cfg_content}
            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)
except Exception as e:
    node.warn(str(e))
"""

    return textwrap.dedent(outer).lstrip()
