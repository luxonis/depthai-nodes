import numpy as np


def get_top_values(heatmap):
    """Get the top values from the heatmap tensor.

    @param heatmap: Heatmap tensor.
    @type heatmap: np.ndarray
    @return: Y and X coordinates of the top values.
    @rtype: Tuple[np.ndarray, np.ndarray]
    """
    batchsize, ny, nx, num_joints = heatmap.shape
    heatmap_flat = heatmap.reshape(batchsize, nx * ny, num_joints)

    heatmap_top = np.argmax(heatmap_flat, axis=1)

    Y, X = (heatmap_top // nx), (heatmap_top % nx)
    return Y, X


def get_pose_prediction(heatmap, locref, scale_factors):
    """Get the pose prediction from the heatmap and locref tensors. Used for SuperAnimal
    model.

    @param heatmap: Heatmap tensor.
    @type heatmap: np.ndarray
    @param locref: Locref tensor.
    @type locref: np.ndarray
    @param scale_factors: Scale factors for the x and y axes.
    @type scale_factors: Tuple[float, float]
    @return: Pose prediction.
    @rtype: np.ndarray
    """
    Y, X = get_top_values(heatmap)
    batch_size, num_joints = X.shape

    heatmap_values = heatmap[0, Y.ravel(), X.ravel(), np.arange(num_joints)].reshape(
        1, 1, num_joints, 1
    )  # Shape: [num_joints, 1] for DZ's 3rd dimension

    if locref is not None:
        locref_updates = locref[
            0, Y.ravel(), X.ravel(), np.arange(num_joints), :
        ].reshape(
            1, 1, num_joints, 2
        )  # Shape: [1, 1, num_joints, 2] for DZ's 1st and 2nd dimensions
    else:
        locref_updates = np.zeros((1, 1, num_joints, 2))

    DZ = np.concatenate(
        (locref_updates, heatmap_values), axis=3
    )  # Concatenate along the last dimension to form [1, 1, num_joints, 3]

    X = X * scale_factors[1] + 0.5 * scale_factors[1] + DZ[:, :, :, 0]
    Y = Y * scale_factors[0] + 0.5 * scale_factors[0] + DZ[:, :, :, 1]

    pose = np.concatenate(
        (np.expand_dims(X, axis=-1), np.expand_dims(Y, axis=-1), DZ[:, :, :, 2:3]),
        axis=3,
    )

    return pose
