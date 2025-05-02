# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from probpose.util import to_numpy


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, f"Invalid shape {heatmaps.shape}"

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.0] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def _calc_distances(
    preds: np.ndarray, gts: np.ndarray, mask: np.ndarray, norm_factor: np.ndarray
) -> np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1
    )
    return distances.T


def _distance_acc(distances: np.ndarray, thr: float = 0.5) -> float:
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        - instance number: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def _get_subpixel_maximums(heatmaps, locs):
    # Extract integer peak locations
    x_locs = locs[:, 0].astype(np.int32)
    y_locs = locs[:, 1].astype(np.int32)

    # Ensure we are not near the boundaries (avoid boundary issues)
    valid_mask = (
        (x_locs > 0)
        & (x_locs < heatmaps.shape[2] - 1)
        & (y_locs > 0)
        & (y_locs < heatmaps.shape[1] - 1)
    )

    # Initialize the output array with the integer locations
    subpixel_locs = locs.copy()

    if np.any(valid_mask):
        # Extract valid locations
        x_locs_valid = x_locs[valid_mask]
        y_locs_valid = y_locs[valid_mask]

        # Compute gradients (dx, dy) and second derivatives (dxx, dyy)
        dx = (
            heatmaps[valid_mask, y_locs_valid, x_locs_valid + 1]
            - heatmaps[valid_mask, y_locs_valid, x_locs_valid - 1]
        ) / 2.0
        dy = (
            heatmaps[valid_mask, y_locs_valid + 1, x_locs_valid]
            - heatmaps[valid_mask, y_locs_valid - 1, x_locs_valid]
        ) / 2.0
        dxx = (
            heatmaps[valid_mask, y_locs_valid, x_locs_valid + 1]
            + heatmaps[valid_mask, y_locs_valid, x_locs_valid - 1]
            - 2 * heatmaps[valid_mask, y_locs_valid, x_locs_valid]
        )
        dyy = (
            heatmaps[valid_mask, y_locs_valid + 1, x_locs_valid]
            + heatmaps[valid_mask, y_locs_valid - 1, x_locs_valid]
            - 2 * heatmaps[valid_mask, y_locs_valid, x_locs_valid]
        )

        # Avoid division by zero by setting a minimum threshold for the second derivatives
        dxx = np.where(dxx != 0, dxx, 1e-6)
        dyy = np.where(dyy != 0, dyy, 1e-6)

        # Calculate the sub-pixel shift
        subpixel_x_shift = -dx / dxx
        subpixel_y_shift = -dy / dyy

        # Update subpixel locations for valid indices
        subpixel_locs[valid_mask, 0] += subpixel_x_shift
        subpixel_locs[valid_mask, 1] += subpixel_y_shift

    return subpixel_locs


def _prepare_oks_kernels(K, H, W, kpt_sigmas: np.ndarray):
    bbox_area = np.sqrt(H / 1.25 * W / 1.25)

    # Generate kernels for all keypoints once for later re-use
    kernels = []
    for k in range(K):
        vars = (kpt_sigmas[k] * 2) ** 2
        s = vars * bbox_area * 2
        s = np.clip(s, 0.55, 3.0)
        radius = np.ceil(s * 3).astype(int)
        diameter = 2 * radius + 1
        diameter = np.ceil(diameter).astype(int)
        # kernel_sizes[kernel_sizes % 2 == 0] += 1
        center = diameter // 2
        dist_x = np.arange(diameter) - center
        dist_y = np.arange(diameter) - center
        dist_x, dist_y = np.meshgrid(dist_x, dist_y)
        dist = np.sqrt(dist_x**2 + dist_y**2)
        oks_kernel = np.exp(-(dist**2) / (2 * s))
        oks_kernel = oks_kernel / oks_kernel.sum()

        oks_kernel = oks_kernel.reshape(1, diameter, diameter)
        kernels.append(oks_kernel)

    return kernels

def pad_scipy_reflect_vectorized(input_tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Manually implements padding equivalent to SciPy's 'reflect' mode using
    vectorized PyTorch operations (slicing, flip, cat).

    SciPy 'reflect': Extends by reflecting about the center of the edge pixels.

    Args:
        input_tensor (torch.Tensor): Input tensor (N, C, H, W).
        padding (tuple[int, int, int, int]): Padding tuple (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        torch.Tensor: Padded tensor.
    """
    N, C, H, W = input_tensor.shape
    pad_left, pad_right, pad_top, pad_bottom = padding

    # --- Vertical padding ---
    if pad_top > 0:
        top_slice = input_tensor[..., 0:pad_top, :]
        top_pad = top_slice.flip(dims=[-2])
    else: top_pad = torch.empty((N, C, 0, W), dtype=input_tensor.dtype, device=input_tensor.device) # Handle 0 padding

    if pad_bottom > 0:
        bottom_slice = input_tensor[..., H-pad_bottom:H, :]
        bottom_pad = bottom_slice.flip(dims=[-2])
    else: bottom_pad = torch.empty((N, C, 0, W), dtype=input_tensor.dtype, device=input_tensor.device)

    vert_padded = torch.cat([top_pad, input_tensor, bottom_pad], dim=-2) # dim=-2 is Height

    # --- Horizontal padding (using the vertically padded tensor) ---
    # Recalculate height for slicing after vertical padding
    H_padded = vert_padded.shape[-2]

    if pad_left > 0:
        left_slice = vert_padded[..., :, 0:pad_left]
        left_pad = left_slice.flip(dims=[-1])
    else: left_pad = torch.empty((N, C, H_padded, 0), dtype=input_tensor.dtype, device=input_tensor.device)

    if pad_right > 0:
        right_slice = vert_padded[..., :, W-pad_right:W]
        right_pad = right_slice.flip(dims=[-1])
    else: right_pad = torch.empty((N, C, H_padded, 0), dtype=input_tensor.dtype, device=input_tensor.device)

    output_padded = torch.cat([left_pad, vert_padded, right_pad], dim=-1) # dim=-1 is Width

    return output_padded


def scipy_convolve2d_reflect_pytorch(input_tensor: torch.Tensor,
                                     kernel_tensor: torch.Tensor,
                                     padding_func=pad_scipy_reflect_vectorized) -> torch.Tensor:
    """
    Performs 2D convolution mimicking scipy.ndimage.convolve(mode='reflect').

    Uses the specified padding function (defaults to vectorized version) and
    flips the kernel for true convolution.

    Args:
        input_tensor (torch.Tensor): Input tensor (N, C, H, W).
        kernel_tensor (torch.Tensor): Kernel tensor (Cout, Cin, Kh, Kw).
        padding_func (callable): The function used for padding.

    Returns:
        torch.Tensor: Output tensor (N, Cout, H, W).
    """
    if not (input_tensor.dim() == 4 and kernel_tensor.dim() == 4):
         raise ValueError("Input and kernel tensors must be 4D (N, C, H, W) and (Cout, Cin, Kh, Kw)")

    N, C, H, W = input_tensor.shape
    Cout, Cin, Kh, Kw = kernel_tensor.shape

    if C != Cin:
         raise ValueError(f"Input channels ({C}) must match kernel input channels ({Cin})")

    # 1. Calculate Padding Amount
    # Handle non-square kernels if necessary, but assume square for simplicity now
    if Kh % 2 == 0 or Kw % 2 == 0:
        print("Warning: Kernel dimensions should be odd for standard centered padding.")
    pad_h = Kh // 2
    pad_w = Kw // 2
    padding_dims = (pad_w, pad_w, pad_h, pad_h) # (left, right, top, bottom)

    # 2. Pad the input tensor using the specified padding function
    input_padded = padding_func(input_tensor, padding_dims)

    # 3. Flip the kernel for true convolution.
    kernel_flipped = kernel_tensor.flip(-1).flip(-2)

    # 4. Apply F.conv2d (cross-correlation) with padding=0.
    output = F.conv2d(input_padded, kernel_flipped, padding=0, stride=1, dilation=1)

    return output


def get_heatmap_expected_value(
    heatmaps: np.ndarray,
    sigmas: np.ndarray,
    parzen_size: float = 0.1,
    return_heatmap: bool = False,
    backend: str = "scipy",
) -> tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, f"Invalid shape {heatmaps.shape}"

    assert parzen_size >= 0.0 and parzen_size <= 1.0, (
        f"Invalid parzen_size {parzen_size}"
    )

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = 1
        FIRST_DIM = K
        heatmaps_flatten = heatmaps.reshape(1, K, H, W)
    else:
        B, K, H, W = heatmaps.shape
        FIRST_DIM = K * B
        heatmaps_flatten = heatmaps.reshape(B, K, H, W)

    KERNELS = _prepare_oks_kernels(K, H, W, sigmas)

    heatmaps_convolved = np.zeros_like(heatmaps_flatten)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    for k in range(K):
        htm_flat = heatmaps_flatten[:, k, :, :].reshape(B, H, W)
        if backend == "torch":
            htm_flat_torch = (
                torch.from_numpy(htm_flat)
                .unsqueeze(1)
                .to(device=device, dtype=torch.float64)
            )  # Shape: (B, 1, H, W)
            kernel = KERNELS[k]
            kernel_torch = (
                torch.from_numpy(kernel)
                .flip(-1)
                .flip(-2)
                .unsqueeze(1)
                .to(device=device, dtype=torch.float64)
            )  # Shape: (1, 1, kernel_size, kernel_size)

         
            result_torch = scipy_convolve2d_reflect_pytorch(
                htm_flat_torch, kernel_torch
            )
            htm_conv = to_numpy(result_torch)
        else:
            from scipy.ndimage import convolve
            htm_conv = convolve(htm_flat, KERNELS[k], mode='reflect').reshape(B, 1, H, W)

        heatmaps_convolved[:, k, :, :] = htm_conv

    heatmaps_convolved = heatmaps_convolved.reshape(B * K, H * W)
    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_convolved, axis=1), shape=(H, W)
    )
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    # Apply mean-shift to get sub-pixel locations
    locs = _get_subpixel_maximums(heatmaps_convolved.reshape(B * K, H, W), locs)

    x_locs_int = np.round(x_locs).astype(int)
    x_locs_int = np.clip(x_locs_int, 0, W - 1)
    y_locs_int = np.round(y_locs).astype(int)
    y_locs_int = np.clip(y_locs_int, 0, H - 1)
    vals = heatmaps_flatten[np.arange(B), np.arange(K), y_locs_int, x_locs_int]

    heatmaps_convolved = heatmaps_convolved.reshape(B, K, H, W)

    if B > 1:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)
        heatmaps_convolved = heatmaps_convolved.reshape(B, K, H, W)
    else:
        locs = locs.reshape(K, 2)
        vals = vals.reshape(K)
        heatmaps_convolved = heatmaps_convolved.reshape(K, H, W)

    if return_heatmap:
        return locs, vals, heatmaps_convolved
    else:
        return locs, vals
