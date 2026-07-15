import numpy as np


def compute_segmentation_class_map(
    segmentation_mask: np.ndarray,
    *,
    classes_in_one_layer: bool = False,
    background_class: bool = False,
) -> np.ndarray:
    """Convert segmentation logits into a class map."""
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
