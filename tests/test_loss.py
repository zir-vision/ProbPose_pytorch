import numpy as np
import torch
from probpose.codec import Codec
from probpose.loss import ProbPoseLoss

def test_loss():
    # Initialize the codec and loss function
    codec = Codec((768, 768), (192, 192), np.array([0.1] * 20))
    loss_fn = ProbPoseLoss(codec)

    # Create dummy data
    batch_size = 2
    num_keypoints = 20
    heatmaps = torch.randn(batch_size, num_keypoints, 256, 256)
    sigmas = torch.randn(num_keypoints)
    gt = {
        "heatmaps": heatmaps,
        "in_image": torch.ones(batch_size, 1, 256, 256),
        "keypoints_visible": torch.ones(batch_size, num_keypoints, 1, 1),
        "keypoints_visibility": torch.ones(batch_size, num_keypoints, 1, 1),
    }

    pred = (
        # Heatmaps
        torch.randn(batch_size, num_keypoints, 256, 256),
        # Probabilities
        torch.randn(batch_size, num_keypoints, 256, 256),
        # Visibilities
        torch.randn(batch_size, num_keypoints, 256, 256),
        # OKS
        torch.randn(batch_size, num_keypoints, 256, 256),
        # Error
        torch.randn(batch_size, num_keypoints, 256, 256),
    )
    # Compute the loss
    loss = loss_fn(gt, pred)
    
    # Check if the loss is a tensor
    assert isinstance(loss["kpt"], torch.Tensor), "Loss should be a tensor"