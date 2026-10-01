import numpy as np


def compute_segmentation_class_map(
    segmentation_mask: np.ndarray,
    *,
    classes_in_one_layer: bool = False,
    background_class: bool = False,
) -> np.ndarray:
    """Convert segmentation scores or encoded labels into a uint8 class map.

    Args:
        segmentation_mask: CHW or HWC tensor, optionally batched. A 4D tensor uses its
            first batch item; the smallest axis is treated as the class axis.
        classes_in_one_layer: Whether a single channel already encodes class IDs rather
            than foreground scores.
        background_class: Replace winning class 0 with 255 for multi-class score
            tensors.

    Returns:
        An HW uint8 label map. Unassigned pixels are 255. A single foreground-score
        channel is compared against zero and emits class 0 for positive pixels.

    Raises:
        ValueError: If the unbatched tensor is not 3D or the resulting values are
            outside [0, 255].
    """
    mask = np.asarray(segmentation_mask)

    if mask.ndim == 4:
        mask = mask[0]

    if mask.ndim != 3:
        raise ValueError(f"Expected 3D output tensor, got {mask.ndim}D.")

    np_function = np.argmax
    mask_shape = mask.shape
    min_dim = np.argmin(mask_shape)
    if min_dim == len(mask_shape) - 1:
        mask = mask.transpose(2, 0, 1)

    adding_unassigned_class = False
    if mask.shape[0] == 1:
        if classes_in_one_layer:
            np_function = np.max
        else:
            adding_unassigned_class = True
            mask = np.vstack(
                (
                    np.zeros((1, mask.shape[1], mask.shape[2]), dtype=np.float32),
                    mask,
                )
            )

    class_map = np_function(mask, axis=0).reshape(mask.shape[1], mask.shape[2])

    if adding_unassigned_class:
        class_map = np.where(class_map == 0, 255, class_map - 1)
    elif background_class and not classes_in_one_layer:
        class_map = np.where(class_map == 0, 255, class_map)

    if np.any(class_map < 0) or np.any(class_map > 255):
        raise ValueError(
            "Segmentation mask values must be in the uint8 range [0, 255]."
        )

    return class_map.astype(np.uint8)
