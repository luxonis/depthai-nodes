from typing import List, Optional


def generate_script_content(
    resize_width: int,
    resize_height: int,
    resize_mode: str = "NONE",
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
        "LETTERBOX", "NONE", "STRETCH". Default: "NONE"
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
            cfg = ImageManipConfigV2()
            rect = RotatedRect()
            rect.center.x = (det.xmin + det.xmax) / 2
            rect.center.y = (det.ymin + det.ymax) / 2
            rect.size.width = det.xmax - det.xmin
            rect.size.height = det.ymax - det.ymin
            rect.size.width = rect.size.width + {padding} * 2
            rect.size.height = rect.size.height + {padding} * 2
            rect.angle = 0

            cfg.addCropRotatedRect(rect, True)
            cfg.setOutputSize({resize_width}, {resize_height}, ImageManipConfigV2.ResizeMode.{resize_mode})
        """
    validate_label = (
        f"""
            if det.label not in {valid_labels}:
                continue
        """
        if valid_labels
        else ""
    )
    return f"""
try:
    while True:
        frame = node.inputs['preview'].get()
        dets = node.inputs['det_in'].get()

        for i, det in enumerate(dets.detections):
            {validate_label.strip()}

            {cfg_content.strip()}

            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)

except Exception as e:
    node.warn(str(e))
"""
