import numpy as np
import torch
from probpose.codec import Codec, ArgMaxProbMap
from probpose.loss import ProbPoseLoss

def test_kpt_loss():
    # Initialize the codec and loss function
    codec = ArgMaxProbMap((768, 768), (192, 192), np.array([0.1] * 20))
    loss_fn = ProbPoseLoss(codec)

    # Create one keypoint in the middle of the image
    encoded = codec.encode(
        np.array([[[96.0, 96.0]]]),  # x, y coordinates
        np.array([[1.0]]),  # visibility
        np.array([[1.0]]),  # probability
    )

    heatmap = encoded["heatmaps"][None, :, :, :]
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap min: {heatmap.min()}, max: {heatmap.max()}")
    loss = loss_fn.keypoint_loss_module(
        torch.from_numpy(np.zeros_like(heatmap)),  # predicted heatmap
        torch.from_numpy(heatmap),  # target heatmap
        torch.tensor([1.0], dtype=torch.float32).unsqueeze(0),  # visibility
    )

    raise ValueError(f"Keypoint loss: {loss.item()}")


