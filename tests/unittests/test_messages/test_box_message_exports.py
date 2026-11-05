import subprocess
import sys

import depthai_nodes
import depthai_nodes.message

BOX_NAMES = [
    "BoxDetections",
    "BoxDetection",
    "BoxMeasurements",
    "BoxMeasurementResult",
    "BoxDimensions",
    "MeasurementStatus",
    "SupportPlane",
    "SupportPlaneCalibrationControl",
    "SupportPlaneCalibrationResult",
    "SupportPlaneVerificationResult",
    "CalibrationOutcome",
]


def test_box_messages_are_exported_from_message_and_top_level():
    for name in BOX_NAMES:
        assert name in depthai_nodes.message.__all__
        assert getattr(depthai_nodes, name) is getattr(depthai_nodes.message, name)


def test_background_label_is_not_exported():
    assert "BACKGROUND_LABEL" not in depthai_nodes.message.__all__
    assert not hasattr(depthai_nodes, "BACKGROUND_LABEL")


def test_box_messages_import_without_pydantic():
    code = (
        "import sys; sys.modules['pydantic'] = None; "
        "from depthai_nodes import *; from depthai_nodes.message import SupportPlane"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
