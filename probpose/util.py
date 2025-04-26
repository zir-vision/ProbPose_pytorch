from dataclasses import dataclass

import numpy as np
from torch import Tensor


def to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy()


@dataclass
class ProbPoseGroundTruth:
    heatmaps: np.ndarray
    in_image: np.ndarray
    # whether the keypoint is annotated, in coco visibility: v == 2
    keypoints_visible: np.ndarray
    # whether the keypoint is visible, in coco visibility: v == 1
    keypoints_visibility: np.ndarray
