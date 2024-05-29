import numpy as np


def unnormalize_image(image):
    """
    Un-normalize an image tensor by scaling it to the [0, 255] range.

    Parameters:
    - image (np.ndarray): The normalized image tensor of shape (H, W, C) or (C, H, W).

    Returns:
    - np.ndarray: The un-normalized image.
    """
    # Normalize the image tensor to the range [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # Scale to [0, 255]
    image = image * 255.0

    # Clip the values to be in the proper range and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image
