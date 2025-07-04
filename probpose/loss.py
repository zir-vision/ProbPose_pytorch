from functools import partial
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from probpose.codec import ArgMaxProbMap, Codec
from probpose.heatmap import (
    _calc_distances,
    _distance_acc,
    get_heatmap_expected_value,
    get_heatmap_maximum,
)
from probpose.util import ProbPoseGroundTruth, to_numpy

class OKSHeatmapLoss(nn.Module):
    """Loss that maximizes expected Object Keypoint Similarity (OKS) score.

    This loss computes the expected OKS by multiplying probability maps with OKS
    maps and maximizes the OKS score by minimizing (1-OKS). The approach was
    introduced in "ProbPose: A Probabilistic Approach to 2D Human Pose Estimation"
    by Purkrabek et al. in 2025.
    See https://arxiv.org/abs/2412.02254 for more details.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 smoothing_weight: float = 0.2, 
                 gaussian_weight: float = 0.0, 
                 loss_weight: float = 1.,
                 oks_type: str = "minus"):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight
        self.smoothing_weight = smoothing_weight
        self.gaussian_weight = gaussian_weight
        self.oks_type = oks_type.lower()

        assert self.oks_type in ["minus", "plus", "both"]

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Tensor | None = None,
                mask: Tensor | None = None,
                per_pixel: bool = False,
                per_keypoint: bool = False) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        # print_tensor_stats(output)
        # print_tensor_stats(target)
        assert target.max() <= 1, 'target should be normalized'
        assert target.min() >= 0, 'target should be normalized'

        B, K, H, W = output.shape

        _mask = self._get_mask(target, target_weights, mask)
        # print(f"{_mask=}")
        oks_minus = output * (1-target)
        oks_plus = (1-output) * (target)
        if self.oks_type == "both":
            oks = (oks_minus + oks_plus) / 2
        elif self.oks_type == "minus":
            oks = oks_minus
        elif self.oks_type == "plus":
            oks = oks_plus
        else:
            raise ValueError(f"oks_type {self.oks_type} not recognized")

        mse = F.mse_loss(output, target, reduction='none')

        # Smoothness loss
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(output.device)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(output.device)
        gradient_x = F.conv2d(output.reshape(B*K, 1, H, W), sobel_x, padding='same')
        gradient_y = F.conv2d(output.reshape(B*K, 1, H, W), sobel_y, padding='same')
        gradient = (gradient_x**2 + gradient_y**2).reshape(B, K, H, W)
        
        if _mask is not None:
            oks = oks * _mask
            mse = mse * _mask
            gradient = gradient * _mask

            
        oks_minus_weight = (
            1 - self.smoothing_weight - self.gaussian_weight
        )
        # print_tensor_stats(oks)
        if per_pixel:
            loss = (
                self.smoothing_weight * gradient +
                oks_minus_weight * oks +
                self.gaussian_weight * mse
            )
        elif per_keypoint:
            max_gradient, _ = gradient.reshape((B, K, H*W)).max(dim=-1)
            loss = (
                oks_minus_weight * oks.sum(dim=(2, 3)) + 
                self.smoothing_weight * max_gradient +
                self.gaussian_weight * mse.mean(dim=(2, 3))
            )
        else:
            max_gradient, _ = gradient.reshape((B, K, H*W)).max(dim=-1)
            loss = (
                oks_minus_weight * oks.sum(dim=(2, 3)) + 
                self.smoothing_weight * max_gradient +
                self.gaussian_weight * mse.mean(dim=(2, 3))
            ).mean()
            
        return loss * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Tensor | None,
                  mask: Tensor | None) -> Tensor | None:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                    f'mask and target have mismatched shapes {mask.shape} v.s.'
                    f'{target.shape}')

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape +
                                        (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask


class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            before output. Defaults to False.
    """

    def __init__(
        self,
        use_target_weight=False,
        loss_weight=1.0,
        reduction="mean",
        use_sigmoid=False,
    ):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), (
            f"the argument "
            f"`reduction` should be either 'mean', 'sum' or 'none', "
            f"but got {reduction}"
        )

        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        criterion = (
            F.binary_cross_entropy
            if use_sigmoid
            else F.binary_cross_entropy_with_logits
        )
        self.criterion = partial(criterion, reduction="none")
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = loss * target_weight
        else:
            loss = self.criterion(output, target)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss * self.loss_weight


class MSELoss(nn.Module):
    """MSE loss for coordinate regression."""

    def __init__(self, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.criterion = F.mse_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            target (torch.Tensor[N, K, 2]): Target regression.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


class L1LogLoss(nn.Module):
    """L1LogLoss loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.criterion = F.smooth_l1_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        # Use logarithm to compute relative error
        output = torch.log(1 + output)
        target = torch.log(1 + target)

        if self.use_target_weight:
            assert target_weight is not None
            assert output.ndim >= target_weight.ndim

            for i in range(output.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


class ProbPoseLoss(nn.Module):
    # TODO: fast_decoder. Argmax thing
    def __init__(self, codec: Codec, freeze_error: bool = True):
        super().__init__()
        self.codec = codec
        # self.keypoint_loss_module = KeypointMSELoss(use_target_weight=True)
        self.keypoint_loss_module = OKSHeatmapLoss(
            use_target_weight=True,
            smoothing_weight=0.05,
            oks_type="minus",
        )
        self.probability_loss_module = BCELoss(use_target_weight=False, use_sigmoid=True)
        self.visibility_loss_module = BCELoss(use_target_weight=False, use_sigmoid=True)
        self.oks_loss_module = MSELoss(use_target_weight=True)
        self.error_loss_module = L1LogLoss(use_target_weight=True)
        self.freeze_error = freeze_error
        self.freeze_oks = False

    def forward(
        self,
        gt: Sequence[dict[str, Tensor]],
        pred: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        keypoint_weights: Tensor | None = None,  # (B, C)
        learn_heatmaps_from_zeros: bool = False,
        compute_acc: bool = False,
    ):
        dt_heatmaps, dt_probs, dt_vis, dt_oks, dt_errs = pred
        device = dt_heatmaps.device
        B, C, H, W = dt_heatmaps.shape

        if keypoint_weights is None:
            keypoint_weights = torch.ones((B, C), device=device, dtype=dt_heatmaps.dtype)

        # Extract GT data
        gt_heatmaps = gt["heatmaps"].to(device, dtype=dt_heatmaps.dtype)
        gt_probs = gt["in_image"].to(device, dtype=int)
        gt_annotated = gt["keypoints_visible"].to(device, dtype=int)
        gt_vis = gt["keypoints_visibility"].to(device, dtype=int)

        # Compute GT errors and OKS
        if self.freeze_error:
            gt_errs = torch.zeros((B, C, 1), device=device, dtype=dt_errs.dtype)
        else:
            gt_errs = self._error_from_heatmaps(gt_heatmaps, dt_heatmaps)
            gt_errs = torch.from_numpy(gt_errs).to(device, dtype=dt_errs.dtype)
        if self.freeze_oks:
            gt_oks = torch.zeros((B, C, 1), device=device, dtype=dt_oks.dtype)
            oks_weight = torch.zeros((B, C, 1), device=device, dtype=dt_oks.dtype)
        else:
            gt_oks, oks_weight = self._oks_from_heatmaps(
                gt_heatmaps,
                dt_heatmaps,
                gt_probs & gt_annotated,
                heatmap_size=(W, H),
            )

        gt_oks = gt_oks.to(device).to(dt_oks.dtype)
        oks_weight = oks_weight.to(device).to(dt_oks.dtype)
        gt_errs = gt_errs.to(device).to(dt_errs.dtype)

        # Reshape everything to comparable shapes
        gt_heatmaps = gt_heatmaps.view((B, C, H, W))
        dt_heatmaps = dt_heatmaps.view((B, C, H, W))
        gt_probs = gt_probs.view((B, C))
        # print(gt_probs)
        dt_probs = dt_probs.view((B, C))
        # print(dt_probs)
        gt_vis = gt_vis.view((B, C))
        dt_vis = dt_vis.view((B, C))
        gt_oks = gt_oks.view((B, C))
        dt_oks = dt_oks.view((B, C))
        gt_errs = gt_errs.view((B, C))
        dt_errs = dt_errs.view((B, C))
        keypoint_weights = keypoint_weights.view((B, C))
        gt_annotated = gt_annotated.view((B, C))
        # oks_weight = oks_weight.view((B, C))

        annotated_in = gt_annotated & (gt_probs > 0.5)

        # calculate losses
        losses = {}
        if learn_heatmaps_from_zeros:
            heatmap_weights = gt_annotated
        else:
            heatmap_weights = keypoint_weights

        heatmap_loss_pxl = self.keypoint_loss_module(
            dt_heatmaps, gt_heatmaps, heatmap_weights, per_pixel=True
        )
        heatmap_loss = heatmap_loss_pxl.mean()
        probability_loss = self.probability_loss_module(
            dt_probs, gt_probs.float()
        )

        # Weight the annotated keypoints such that sum of weights of invisible keypoints is the same as visible ones
        invisible_in = (gt_vis == 0) & (gt_annotated > 0.5)
        visible_in = (gt_vis > 0) & (gt_annotated > 0.5)
        weighted_annotated_in: Tensor = annotated_in.clone().to(float)
        weighted_annotated_in[invisible_in] = (1 / (invisible_in.sum() + 1e-10)).to(
            weighted_annotated_in.dtype
        )
        weighted_annotated_in[visible_in] = (1 / (visible_in.sum() + 1e-10)).to(
            weighted_annotated_in.dtype
        )
        weighted_annotated_in = (
            weighted_annotated_in
            / weighted_annotated_in[weighted_annotated_in > 0].min()
        )
        weighted_annotated_in = weighted_annotated_in.to(dt_vis.dtype)

        visibility_loss = self.visibility_loss_module(
            dt_vis, gt_vis.float(), weighted_annotated_in
        )
        oks_loss = self.oks_loss_module(dt_oks, gt_oks, annotated_in)
        error_loss = self.error_loss_module(dt_errs, gt_errs, annotated_in)

        losses.update(
            kpt=heatmap_loss,
            probability=probability_loss,
            visibility=visibility_loss,
            oks=oks_loss,
            error=error_loss,
        )

        # calculate accuracy
        if compute_acc:
            acc_pose = self.get_pose_accuracy(
                dt_heatmaps, gt_heatmaps, keypoint_weights > 0.5
            )

            # Calculate the best binary accuracy for probability
            acc_prob, _ = self.get_binary_accuracy(
                dt_probs,
                gt_probs,
                gt_annotated > 0.5,
                force_balanced=True,
            )

            # Calculate the best binary accuracy for visibility
            acc_vis, _ = self.get_binary_accuracy(
                dt_vis,
                gt_vis,
                annotated_in > 0.5,
                force_balanced=True,
            )

            # Calculate the MAE for OKS
            acc_oks = self.get_mae(
                dt_oks,
                gt_oks,
                annotated_in > 0.5,
            )

            # Calculate the MAE for euclidean error
            acc_err = self.get_mae(
                dt_errs,
                gt_errs,
                annotated_in > 0.5,
            )

            return losses, {
                "kpt": acc_pose,
                "probability": acc_prob,
                "visibility": acc_vis,
                "oks": acc_oks,
                "error": acc_err,
            }

        return losses

    def _error_from_heatmaps(self, gt_heatmaps: Tensor, dt_heatmaps: Tensor) -> Tensor:
        """Calculate the error from heatmaps.

        Args:
            heatmaps (Tensor): The predicted heatmaps.

        Returns:
            Tensor: The predicted error.
        """
        # Transform to numpy
        gt_heatmaps = to_numpy(gt_heatmaps)
        dt_heatmaps = to_numpy(dt_heatmaps)

        # Get locations from heatmaps
        B, C, H, W = gt_heatmaps.shape
        gt_coords = np.zeros((B, C, 2))
        dt_coords = np.zeros((B, C, 2))
        for i, (gt_htm, dt_htm) in enumerate(zip(gt_heatmaps, dt_heatmaps)):
            # coords, score = self.fast_decoder.decode(gt_htm)
            coords, score = self.codec.decode_heatmap(gt_htm)
            coords = coords.squeeze()
            gt_coords[i, :, :] = coords

            # coords, score = self.fast_decoder.decode(dt_htm)
            coords, score = self.codec.decode_heatmap(dt_htm)
            coords = coords.squeeze()
            dt_coords[i, :, :] = coords

        # NaN coordinates mean empty heatmaps -> set them to -1
        # as the error will be ignored by weight
        gt_coords[np.isnan(gt_coords)] = -1

        # Calculate the error
        target_errors = np.linalg.norm(gt_coords - dt_coords, axis=2)
        assert (target_errors >= 0).all(), "Euclidean distance cannot be negative"

        return target_errors

    def _oks_from_heatmaps(
        self,
        gt_heatmaps: Tensor,
        dt_heatmaps: Tensor,
        weight: Tensor,
        heatmap_size: Sequence[int] = (48, 64),
    ) -> Tensor:
        """Calculate the OKS from heatmaps.

        Args:
            heatmaps (Tensor): The predicted heatmaps.

        Returns:
            Tensor: The predicted OKS.
        """
        C = dt_heatmaps.shape[1]

        # Transform to numpy
        gt_heatmaps = to_numpy(gt_heatmaps)
        dt_heatmaps = to_numpy(dt_heatmaps)
        B, C, H, W = gt_heatmaps.shape
        weight = to_numpy(weight).squeeze().reshape((B, C, 1))

        # Get locations from heatmaps
        gt_coords = np.zeros((B, C, 2))
        dt_coords = np.zeros((B, C, 2))
        for i, (gt_htm, dt_htm) in enumerate(zip(gt_heatmaps, dt_heatmaps)):
            # coords, score = self.fast_decoder.decode(gt_htm)
            coords, score = self.codec.decode_heatmap(gt_htm)
            coords = coords.squeeze()
            gt_coords[i, :, :] = coords

            # coords, score = self.fast_decoder.decode(dt_htm)
            coords, score = self.codec.decode_heatmap(dt_htm)
            coords = coords.squeeze()
            dt_coords[i, :, :] = coords

        # NaN coordinates mean empty heatmaps -> set them to 0
        gt_coords[np.isnan(gt_coords)] = 0

        # Add probability as visibility
        gt_coords = gt_coords * weight
        dt_coords = dt_coords * weight
        gt_coords = np.concatenate((gt_coords, weight * 2), axis=2)
        dt_coords = np.concatenate((dt_coords, weight * 2), axis=2)

        # Calculate the oks
        target_oks = []
        oks_weights = []
        for i in range(len(gt_coords)):
            gt_kpts = gt_coords[i]
            dt_kpts = dt_coords[i]
            valid_gt_kpts = gt_kpts[:, 2] > 0
            if not valid_gt_kpts.any():
                # Changed for per-keypoint OKS
                target_oks.append(np.zeros(C))
                oks_weights.append(0)
                continue

            gt_bbox = np.array(
                [
                    0,
                    0,
                    heatmap_size[1],
                    heatmap_size[0],
                ]
            )
            gt = {
                "keypoints": gt_kpts,
                "bbox": gt_bbox,
                "area": gt_bbox[2] * gt_bbox[3],
            }
            dt = {
                "keypoints": dt_kpts,
                "bbox": gt_bbox,
                "area": gt_bbox[2] * gt_bbox[3],
            }
            # Changed for per-keypoint OKS
            oks = compute_oks(
                gt, dt, sigmas=self.codec.probmap.sigmas, use_area=False, per_kpt=True
            )
            target_oks.append(oks)
            oks_weights.append(1)

        target_oks = np.array(target_oks)
        target_oks = torch.from_numpy(target_oks).float()

        oks_weights = np.array(oks_weights)
        oks_weights = torch.from_numpy(oks_weights).float()

        return target_oks, oks_weights

    def get_pose_accuracy(self, dt, gt, mask):
        """Calculate the accuracy of predicted pose."""
        _, avg_acc, _ = pose_pck_accuracy(
            output=to_numpy(dt),
            target=to_numpy(gt),
            mask=to_numpy(mask),
            method="argmax",
        )
        acc_pose = torch.tensor(avg_acc, device=gt.device)
        return acc_pose

    def get_binary_accuracy(self, dt, gt, mask, force_balanced=False):
        """Calculate the binary accuracy."""
        assert dt.shape == gt.shape
        device = gt.device
        dt = to_numpy(dt)
        gt = to_numpy(gt)
        mask = to_numpy(mask)

        dt = dt[mask]
        gt = gt[mask]
        gt = gt.astype(bool)

        if force_balanced:
            # Force the number of positive and negative samples to be balanced
            pos_num = np.sum(gt)
            neg_num = len(gt) - pos_num
            num = min(pos_num, neg_num)
            if num == 0:
                return torch.tensor([0.0], device=device), torch.tensor(
                    [0.0], device=device
                )
            pos_idx = np.where(gt)[0]
            neg_idx = np.where(~gt)[0]

            # Randomly sample the same number of positive and negative samples
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            idx = np.concatenate([pos_idx[:num], neg_idx[:num]])
            dt = dt[idx]
            gt = gt[idx]

        n_samples = len(gt)
        thresholds = np.arange(0.1, 1.0, 0.05)
        preds = dt[:, None] > thresholds
        correct = preds == gt[:, None]
        counts = correct.sum(axis=0)

        # Find the threshold that maximizes the accuracy
        best_idx = np.argmax(counts)
        best_threshold = thresholds[best_idx]
        best_acc = counts[best_idx] / n_samples

        best_acc = torch.tensor(best_acc, device=device).float()
        best_threshold = torch.tensor(best_threshold, device=device).float()
        return best_acc, best_threshold

    def get_mae(self, dt, gt, mask):
        """Calculate the mean absolute error."""
        assert dt.shape == gt.shape
        device = gt.device
        dt = to_numpy(dt)
        gt = to_numpy(gt)
        mask = to_numpy(mask)

        dt = dt[mask]
        gt = gt[mask]
        mae = np.abs(dt - gt).mean()

        mae = torch.tensor(mae, device=device)
        return mae


def compute_oks(gt, dt, sigmas: np.ndarray, use_area=True, per_kpt=False):
    vars = (sigmas * 2) ** 2
    k = len(sigmas)

    def visibility_condition(x):
        return x > 0

    g = np.array(gt["keypoints"]).reshape(k, 3)
    xg = g[:, 0]
    yg = g[:, 1]
    vg = g[:, 2]
    k1 = np.count_nonzero(visibility_condition(vg))
    bb = gt["bbox"]
    x0 = bb[0] - bb[2]
    x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]
    y1 = bb[1] + bb[3] * 2

    d = np.array(dt["keypoints"]).reshape((k, 3))
    xd = d[:, 0]
    yd = d[:, 1]

    if k1 > 0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg

    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
        dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)

    if use_area:
        e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
    else:
        tmparea = gt["bbox"][3] * gt["bbox"][2] * 0.53
        e = (dx**2 + dy**2) / vars / (tmparea + np.spacing(1)) / 2

    if per_kpt:
        oks = np.exp(-e)
        if k1 > 0:
            oks[~visibility_condition(vg)] = 0

    else:
        if k1 > 0:
            e = e[visibility_condition(vg)]
        oks = np.sum(np.exp(-e)) / e.shape[0]

    return oks


def pose_pck_accuracy(
    output: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    thr: float = 0.05,
    normalize: np.ndarray | None = None,
    method: str = "argmax",
) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from heatmaps.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
        target (np.ndarray[N, K, H, W]): Groundtruth heatmaps.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    method = method.lower()
    if method not in ["argmax", "expected"]:
        raise ValueError(f"Invalid method: {method}")

    N, K, H, W = output.shape
    if K == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    if method == "argmax":
        pred, _ = get_heatmap_maximum(output)
        gt, _ = get_heatmap_maximum(target)
    else:
        pred, _ = get_heatmap_expected_value(output)
        gt, _ = get_heatmap_expected_value(target)
    return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)


def keypoint_pck_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    thr: np.ndarray,
    norm_factor: np.ndarray,
) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, norm_factor)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt
