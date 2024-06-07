import numpy as np
from ...messages import MatchedPoints

def create_matched_points_message(reference_points: np.ndarray, target_points: np.ndarray) -> MatchedPoints:
    """
    Create a message for the matched points. The message contains the reference and target points.

    Args:
        reference_points (np.ndarray): Reference points of shape (N,2) meaning [...,[x, y],...].
        target_points (np.ndarray): Target points of shape (N,2) meaning [...,[x, y],...].

    Returns:
        MatchedPoints: Message containing the reference and target points.
    """


    if not isinstance(reference_points, np.ndarray):
        raise ValueError(f"reference_points should be numpy array, got {type(reference_points)}.")
    if len(reference_points.shape) != 2:
        raise ValueError(f"reference_points should be of shape (N,2) meaning [...,[x, y],...], got {reference_points.shape}.")
    if reference_points.shape[1] != 2:
        raise ValueError(f"reference_points 2nd dimension should be of size 2 e.g. [x, y], got {reference_points.shape[1]}.")
    if not isinstance(target_points, np.ndarray):
        raise ValueError(f"target_points should be numpy array, got {type(target_points)}.")
    if len(target_points.shape) != 2:
        raise ValueError(f"target_points should be of shape (N,2) meaning [...,[x, y],...], got {target_points.shape}.")
    if target_points.shape[1] != 2:
        raise ValueError(f"target_points 2nd dimension should be of size 2 e.g. [x, y], got {target_points.shape[1]}.")
    
    matched_points_msg = MatchedPoints()
    matched_points_msg.reference_points = reference_points.tolist()
    matched_points_msg.target_points = target_points.tolist() 

    return matched_points_msg