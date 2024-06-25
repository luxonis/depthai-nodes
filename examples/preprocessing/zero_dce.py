import numpy as np

def preprocess_zero_dce(img):
    # HWC to CHW
    img = np.transpose(img, (2,0,1))

    return img