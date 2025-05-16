# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from probpose.util import to_numpy

try:
    from sparsemax import Sparsemax

    sparsemax_available = True
except ImportError:
    sparsemax_available = False


class ProbMapHead(nn.Module):
    """Multi-variate head predicting all information about keypoints. Apart
    from the heatmap, it also predicts:
        1) Heatmap for each keypoint
        2) Probability of keypoint being in the heatmap
        3) Visibility of each keypoint
        4) Predicted OKS per keypoint
        5) Predictd euclidean error per keypoint
    The heatmap predicting part is the same as HeatmapHead introduced in
    in `Simple Baselines`_ by Xiao et al (2018).

        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer_kernel_size (int | None): Kernel size of the final Conv2d layer.
            Defaults to ``1``
        keypoint_loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        probability_loss (Config): Config of the probability loss. Defaults to use
            :class:`BCELoss`
        visibility_loss (Config): Config of the visibility loss. Defaults to use
            :class:`BCELoss`
        oks_loss (Config): Config of the oks loss. Defaults to use
            :class:`MSELoss`
        error_loss (Config): Config of the error loss. Defaults to use
            :class:`L1LogLoss`
        normalize (float | None): Whether to normalize values in the heatmaps between
            0 and 1 with sigmoid. Defaults to ``False``. pytorch port ???
        detach_probability (bool): Whether to detach the probability
            from gradient computation. Defaults to ``True``
        detach_visibility (bool): Whether to detach the visibility
            from gradient computation. Defaults to ``True``
        freeze_heatmaps (bool): Whether to freeze the heatmaps prediction.
            Defaults to ``False``
        freeze_probability (bool): Whether to freeze the probability prediction.
            Defaults to ``False``
        freeze_visibility (bool): Whether to freeze the visibility prediction.
            Defaults to ``False``
        freeze_oks (bool): Whether to freeze the oks prediction.
            Defaults to ``False``
        freeze_error (bool): Whether to freeze the error prediction.
            Defaults to ``False``
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``


    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        alt_head_kernel_sizes: Sequence[int],
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        conv_out_channels=None,
        conv_kernel_sizes=None,
        final_layer_kernel_size=1,
        normalize: float | None = None,
        detach_probability: bool = True,
        detach_visibility: bool = True,
        freeze_heatmaps: bool = False,
        freeze_probability: bool = False,
        freeze_visibility: bool = False,
        freeze_oks: bool = False,
        freeze_error: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.temperature = 0.5

        self.gauss_sigma = 2.0
        self.gauss_kernel_size = int(2.0 * 3.0 * self.gauss_sigma + 1.0)
        ts = torch.linspace(
            -self.gauss_kernel_size // 2,
            self.gauss_kernel_size // 2,
            self.gauss_kernel_size,
        )
        gauss = torch.exp(-((ts / self.gauss_sigma) ** 2) / 2)
        gauss = gauss / gauss.sum()
        self.gauss_kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)

        self.nonlinearity = nn.ReLU(inplace=True)
        self._build_heatmap_head(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer_kernel_size=final_layer_kernel_size,
            normalize=normalize,
            freeze=freeze_heatmaps,
        )

        self.normalize = normalize

        self.detach_probability = detach_probability
        self._build_probability_head(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=alt_head_kernel_sizes,
            freeze=freeze_probability,
        )

        self.detach_visibility = detach_visibility
        self._build_visibility_head(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=alt_head_kernel_sizes,
            freeze=freeze_visibility,
        )

        self._build_oks_head(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=alt_head_kernel_sizes,
            freeze=freeze_oks,
        )
        self.freeze_oks = freeze_oks

        self._build_error_head(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=alt_head_kernel_sizes,
            freeze=freeze_error,
        )
        self.freeze_error = freeze_error

        self._initialize_weights()

    def _freeze_all_but_temperature(self):
        for param in self.parameters():
            param.requires_grad = False
        self.temperature.requires_grad = True

    def _build_heatmap_head(
        self,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: Sequence[int],
        deconv_kernel_sizes: Sequence[int],
        conv_out_channels: Sequence[int] | None,
        conv_kernel_sizes: Sequence[int] | None,
        final_layer_kernel_size: int | None = 1,
        normalize: float = None,
        freeze: bool = False,
    ) -> nn.Module:
        """Build the heatmap head module."""
        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                deconv_kernel_sizes
            ):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    "be integer sequences with the same length. Got "
                    f"mismatched lengths {deconv_out_channels} and "
                    f"{deconv_kernel_sizes}"
                )

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                conv_kernel_sizes
            ):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    "be integer sequences with the same length. Got "
                    f"mismatched lengths {conv_out_channels} and "
                    f"{conv_kernel_sizes}"
                )

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes,
            )
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer_kernel_size is not None:
            self.final_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=final_layer_kernel_size,
                padding=final_layer_kernel_size // 2,
            )
        else:
            self.final_layer = nn.Identity()

        if normalize is None:
            self.normalize_layer = nn.Identity()
        else:
            if sparsemax_available:
                self.normalize_layer = Sparsemax(dim=-1)
            else:
                raise ImportError(
                    "Sparsemax is not installed. Please install sparsemax to use this feature."
                )

        if freeze:
            for param in self.deconv_layers.parameters():
                param.requires_grad = False
            for param in self.conv_layers.parameters():
                param.requires_grad = False
            for param in self.final_layer.parameters():
                param.requires_grad = False

    def _build_probability_head(
        self, in_channels: int, out_channels: int, kernel_sizes, freeze: bool = False
    ) -> nn.Module:
        """Build the probability head module."""
        ppb_layers = []
        for i in range(len(kernel_sizes)):
            ppb_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            ppb_layers.append(nn.BatchNorm2d(num_features=in_channels))
            ppb_layers.append(
                nn.MaxPool2d(
                    kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0
                )
            )
            ppb_layers.append(self.nonlinearity)
        ppb_layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        ppb_layers.append(nn.Sigmoid())
        self.probability_layers = nn.Sequential(*ppb_layers)

        if freeze:
            for param in self.probability_layers.parameters():
                param.requires_grad = False

    def _build_visibility_head(
        self, in_channels: int, out_channels: int, kernel_sizes, freeze: bool = False
    ) -> nn.Module:
        """Build the visibility head module."""
        vis_layers = []
        for i in range(len(kernel_sizes)):
            vis_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            vis_layers.append(nn.BatchNorm2d(num_features=in_channels))
            vis_layers.append(
                nn.MaxPool2d(
                    kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0
                )
            )
            vis_layers.append(self.nonlinearity)
        vis_layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        vis_layers.append(nn.Sigmoid())
        self.visibility_layers = nn.Sequential(*vis_layers)

        if freeze:
            for param in self.visibility_layers.parameters():
                param.requires_grad = False

    def _build_oks_head(
        self, in_channels: int, out_channels: int, kernel_sizes, freeze: bool = False
    ) -> nn.Module:
        """Build the oks head module."""
        oks_layers = []
        for i in range(len(kernel_sizes)):
            oks_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            oks_layers.append(nn.BatchNorm2d(num_features=in_channels))
            oks_layers.append(
                nn.MaxPool2d(
                    kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0
                )
            )
            oks_layers.append(self.nonlinearity)
        oks_layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        oks_layers.append(nn.Sigmoid())
        self.oks_layers = nn.Sequential(*oks_layers)

        if freeze:
            for param in self.oks_layers.parameters():
                param.requires_grad = False

    def _build_error_head(
        self, in_channels: int, out_channels: int, kernel_sizes, freeze: bool = False
    ) -> nn.Module:
        """Build the error head module."""
        error_layers = []
        for i in range(len(kernel_sizes)):
            error_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            error_layers.append(nn.BatchNorm2d(num_features=in_channels))
            error_layers.append(
                nn.MaxPool2d(
                    kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0
                )
            )
            error_layers.append(self.nonlinearity)
        error_layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        error_layers.append(self.nonlinearity)
        self.error_layers = nn.Sequential(*error_layers)

        if freeze:
            for param in self.error_layers.parameters():
                param.requires_grad = False

    def _make_conv_layers(
        self,
        in_channels: int,
        layer_out_channels: Sequence[int],
        layer_kernel_sizes: Sequence[int],
    ) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(self.nonlinearity)
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(
        self,
        in_channels: int,
        layer_out_channels: Sequence[int],
        layer_kernel_sizes: Sequence[int],
    ) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(
                    f"Unsupported kernel size {kernel_size} for"
                    "deconvlutional layers in "
                    f"{self.__class__.__name__}"
                )

            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(self.nonlinearity)
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize the weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward the network. The input is multi scale feature maps and the
        output is (1) the heatmap, (2) probability, (3) visibility, (4) oks and (5) error.

        Args:
            feats (Tensor): Multi scale feature maps.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: outputs.
        """

        heatmaps = self.forward_heatmap(x)
        probabilities = self.forward_probability(x)
        visibilities = self.forward_visibility(x)
        oks = self.forward_oks(x)
        error = self.forward_error(x)

        # print("Head forward:")
        # print(f"heatmaps: {heatmaps.shape}")
        # print(f"probabilities: {probabilities.shape}")
        # print(f"visibilities: {visibilities.shape}")
        # print(f"oks: {oks.shape}")
        # print(f"error: {error.shape}")

        return heatmaps, probabilities, visibilities, oks, error

    def forward_heatmap(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        B, C, H, W = x.shape
        x = x.reshape((B, C, H * W))
        x = self.normalize_layer(x / self.temperature)
        if self.normalize is not None:
            x = x * self.normalize
        x = torch.clamp(x, 0, 1)
        x = x.reshape((B, C, H, W))

        return x

    def forward_probability(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the probability.

        Args:
            x (Tensor): Multi scale feature maps.
            detach (bool): Whether to detach the probability from gradient

        Returns:
            Tensor: output probability.
        """
        if self.detach_probability:
            x = x.detach()
        x = self.probability_layers(x)
        return x

    def forward_visibility(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the visibility.

        Args:
            x (Tensor): Multi scale feature maps.
            detach (bool): Whether to detach the visibility from gradient

        Returns:
            Tensor: output visibility.
        """
        if self.detach_visibility:
            x = x.detach()
        x = self.visibility_layers(x)
        return x

    def forward_oks(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the oks.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output oks.
        """
        x = x.detach()
        x = self.oks_layers(x)
        return x

    def forward_error(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the euclidean error.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output error.
        """
        x = x.detach()
        x = self.error_layers(x)
        return x
