from typing import List, Optional


def generate_script_content(
    resize_width: int,
    resize_height: int,
    resize_mode: str = "STRETCH",
    padding: float = 0,
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

    cfg_content = f"""
                cfg = ImageManipConfig()
                rect = RotatedRect()

                # pixel-aware minimum in normalized coords
                fw = float(frame.getWidth())
                fh = float(frame.getHeight())
                eps = max(2.0/fw, 2.0/fh)
                margin = 0.5*eps
                pad = {max(0.0, float(padding))}

                # read + order
                xmin = det.xmin; ymin = det.ymin; xmax = det.xmax; ymax = det.ymax
                if xmin > xmax: xmin, xmax = xmax, xmin
                if ymin > ymax: ymin, ymax = ymax, ymin

                # clamp into (0,1) with margin
                if xmin < margin: xmin = margin
                if ymin < margin: ymin = margin
                if xmax > 1.0 - margin: xmax = 1.0 - margin
                if ymax > 1.0 - margin: ymax = 1.0 - margin

                # ensure >= eps in both axes
                if (xmax - xmin) < eps:
                    c = 0.5*(xmin + xmax); xmin = c - 0.5*eps; xmax = c + 0.5*eps
                if (ymax - ymin) < eps:
                    c = 0.5*(ymin + ymax); ymin = c - 0.5*eps; ymax = c + 0.5*eps

                # keep inside after expansion
                if xmin < margin: xmin = margin; xmax = xmin + eps
                if xmax > 1.0 - margin: xmax = 1.0 - margin; xmin = xmax - eps
                if ymin < margin: ymin = margin; ymax = ymin + eps
                if ymax > 1.0 - margin: ymax = 1.0 - margin; ymin = ymax - eps

                # apply padding, clamp, and re-ensure eps
                xmin -= pad; ymin -= pad; xmax += pad; ymax += pad
                if xmin < margin: xmin = margin
                if ymin < margin: ymin = margin
                if xmax > 1.0 - margin: xmax = 1.0 - margin
                if ymax > 1.0 - margin: ymax = 1.0 - margin
                if (xmax - xmin) < eps: xmax = min(1.0 - margin, xmin + eps)
                if (ymax - ymin) < eps: ymax = min(1.0 - margin, ymin + eps)

                # build rect
                rect.center.x = (xmin + xmax) / 2
                rect.center.y = (ymin + ymax) / 2
                rect.size.width  = xmax - xmin
                rect.size.height = ymax - ymin
                rect.angle = 0

                cfg.addCropRotatedRect(rect, True)
                cfg.setOutputSize({resize_width}, {resize_height}, ImageManipConfig.ResizeMode.{resize_mode})
            """

    validate_label = (
        f"            if det.label not in {valid_labels}: continue\n"
        if valid_labels is not None else ""
    )

    return f"""
    try:
        while True:
            frame = node.inputs['preview'].get()
            dets  = node.inputs['det_in'].get()

            for det in dets.detections:
                {validate_label.strip()} 

                {cfg_content.strip()}
                node.outputs['manip_cfg'].send(cfg)
                node.outputs['manip_img'].send(frame)
    except Exception as e:
        node.warn(str(e))
    """
