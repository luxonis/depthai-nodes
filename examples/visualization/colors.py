def get_adas_colors():
    colors = [
        (0, 0, 0),  # class 0 - black
        (128, 0, 0),  # class 1 - maroon
        (0, 128, 0),  # class 2 - green
        (128, 128, 0),  # class 3 - olive
        (0, 0, 128),  # class 4 - navy
        (128, 0, 128),  # class 5 - purple
        (0, 128, 128),  # class 6 - teal
        (128, 128, 128),  # class 7 - gray
        (64, 0, 0),  # class 8 - maroon
        (192, 0, 0),  # class 9 - red
        (64, 128, 0),  # class 10 - olive
        (192, 128, 0),  # class 11 - yellow
        (64, 0, 128),  # class 12 - navy
        (192, 0, 128),  # class 13 - fuchsia
        (64, 128, 128),  # class 14 - aqua
        (192, 128, 128),  # class 15 - silver
        (0, 64, 0),  # class 16 - green
        (128, 64, 0),  # class 17 - orange
        (0, 192, 0),  # class 18 - lime
        (128, 192, 0),  # class 19 - yellow
        (0, 64, 128),  # class 20 - blue
    ]
    return colors


def get_selfie_colors():
    colors = [(0, 0, 0), (0, 255, 0)]
    return colors


def get_ewasr_colors():
    colors = [
        (0, 255, 255),  # class 0 - black
        (255, 255, 0),  # class 1 - maroon
        (0, 0, 0),  # class 2 - green
    ]
    return colors


def get_yolo_colors():
    colors = [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ]

    return colors
