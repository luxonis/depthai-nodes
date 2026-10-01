import numpy as np

from .denormalize import unnormalize_image


def compute_image_output(output_image: np.ndarray) -> np.ndarray:
    """Convert a model image tensor to an 8-bit image array.

    Args:
        output_image: CHW or HWC tensor, optionally preceded by a singleton batch
            dimension.

    Returns:
        A uint8 array with the same channel layout as the unbatched input. Values are
        min-max scaled to [0, 255]; constant tensors become zero.

    Raises:
        ValueError: If the tensor is not 3D after removing a singleton batch dimension.
    """
    image = np.asarray(output_image)

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    if image.ndim != 3:
        raise ValueError(f"Expected 3D output tensor, got {image.ndim}D.")

    return unnormalize_image(image)
