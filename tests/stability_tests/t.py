import pickle

import numpy as np

data = {}
bboxes = np.array([[0, 0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]])
confidences = np.array([0.9, 0.8])

data["bboxes"] = bboxes
data["scores"] = confidences

with open("nn_datas/DetectionParser/mobile-object-localizer.pkl", "wb") as f:
    pickle.dump(data, f)

with open("nn_datas/SCRFDParser/scrfd-face-detection_output.pkl", "rb") as f:
    data = pickle.load(f)

print(data)

data = {
    "model": "luxonis/mobile-object-localizer:192x192",
    "parser": "DetectionParser",
    "detections": [
        {
            "confidence": 0.9,
            "label": -1,
            "x_center": 0.1,
            "y_center": 0.1,
            "width": 0.2,
            "height": 0.2,
            "angle": 0,
        },
        {
            "confidence": 0.8,
            "label": -1,
            "x_center": 0.6,
            "y_center": 0.6,
            "width": 0.2,
            "height": 0.2,
            "angle": 0,
        },
    ],
}

with open("nn_datas/DetectionParser/mobile-object-localizer_output.pkl", "wb") as f:
    pickle.dump(data, f)
