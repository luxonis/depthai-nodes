import cv2

def save_zero_dce_output(output):
    cv2.imwrite(f"zero_dce_output.png", output.getFrame())