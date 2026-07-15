import numpy as np

from .denormalize import unnormalize_image


def compute_image_output(output_image: np.ndarray) -> np.ndarray:
    """Convert a model image output tensor into an image array."""
    image = np.asarray(output_image)

    if image.shape[0] == 1:
        image = image[0]

    if image.ndim != 3:
        raise ValueError(f"Expected 3D output tensor, got {image.ndim}D.")

    return unnormalize_image(image)
