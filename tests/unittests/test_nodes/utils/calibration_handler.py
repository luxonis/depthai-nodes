import os

import depthai as dai


def get_calibration_handler():
    calibration_handler_json_path = os.path.join(
        os.path.dirname(__file__), "calib.json"
    )
    calibration_handler = dai.CalibrationHandler(calibration_handler_json_path)
    return calibration_handler
