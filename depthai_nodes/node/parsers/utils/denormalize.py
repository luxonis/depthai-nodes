import numpy as np


def unnormalize_image(image, normalize=True):
    """Un-normalize an image tensor by scaling it to the [0, 255] range.

    @param image: The normalized image tensor of shape (H, W, C) or (C, H, W).
    @type image: np.ndarray
    @param normalize: Whether to normalize the image tensor. Defaults to True.
    @type normalize: bool
    @return: The un-normalized image.
    @rtype: np.ndarray
    """
    # Normalize the image tensor to the range [0, 1]
    if normalize:
        min_val = image.min()
        max_val = image.max()
        if max_val != min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            # Handle the case where all values in the image are the same
            image = image - min_val

    # Scale to [0, 255] and clip the values to be in the proper range
    image = image * 255.0
    image = np.clip(image, 0, 255)

    # Convert to uint8
    image = image.astype(np.uint8)

    return image
