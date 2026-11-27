"""Associated corrected depth and provenance for one Python application runtime."""

from typing import TYPE_CHECKING

import depthai as dai

if TYPE_CHECKING:
    from depthai_nodes.utils.axial_depth import AxialDepthProvenance


class AxialDepthFrameResult(dai.Buffer):
    """Carry native RAW16 depth and its explicit axial convention together.

    ``frame`` can feed native PointCloud. ``provenance.to_dict()`` can be saved in
    benchmark reports. Python attributes stay in the host runtime, including an
    OAK-4 application's Python process. They do not serialize through XLink or
    survive native ``ImgFrame.clone()``. Retain this result or its provenance when
    passing the native frame through another correction boundary.

    Args:
        frame: Independent corrected or explicitly passed-through depth frame.
        provenance: Policy, geometry identity, and source association.
    """

    def __init__(self, frame: dai.ImgFrame, provenance: "AxialDepthProvenance") -> None:
        super().__init__()
        self.frame = frame
        self.provenance = provenance
        self.setBufferMetadataFrom(frame)
