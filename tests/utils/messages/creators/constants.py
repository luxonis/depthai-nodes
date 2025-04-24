import numpy as np

CLASSES = ["class1", "class2", "class3"]
SCORES = [0.0, 0.25, 0.75]

CLASSIFICATION = {
    "classes": CLASSES,
    "scores": SCORES,
    "sequence_num": 3,
}

COLLECTIONS = {
    "clusters": [
        [[0.0, 0.0], [0.1, 0.1]],
        [[0.2, 0.2], [0.3, 0.3]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
    ],  # two 2-point clusters, and one 3-point cluster
    "lines": np.array(
        [
            [0.0, 0.0, 0.1, 0.1],
            [0.2, 0.2, 0.3, 0.3],
            [0.4, 0.4, 0.5, 0.5],
        ]
    ),  # three lines
}

DETECTIONS = {
    "bboxes": np.array(
        [
            [0.00, 0.20, 0.00, 0.20],
            [0.20, 0.40, 0.20, 0.40],
            [0.40, 0.60, 0.40, 0.60],
        ]
    ),  # three bboxes
    "angles": np.array([0.0, 0.25, 0.75]),
    "labels": np.array([i for i in range(len(CLASSES))]),
    "scores": np.array(SCORES),
    "keypoints": np.array(
        [
            [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]],
            [[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
            [[0.6, 0.6], [0.7, 0.7], [0.8, 0.8]],
        ]
    ),  # three keypoints for each bbox detection
    "keypoints_scores": np.array(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
        ]
    ),
}

HEIGHT, WIDTH = 5, 5
MAX_VALUE = 50
ARRAYS = {
    "2d": np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH)),  # e.g. mask
    "3d": np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH, 3)),  # e.g. image
}
