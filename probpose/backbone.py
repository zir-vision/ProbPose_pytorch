import torch


class RadioBackbone(torch.nn.Module):
    def __init__(self, version: str, mlp: torch.nn.Module | None = None):
        super().__init__()
        self.model = torch.hub.load(
            "NVlabs/RADIO", "radio_model", version=version
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary, positonal_features = self.model(x, feature_fmt="NCHW")
        N, C, H, W = positonal_features.shape
        if self.mlp is not None:
            positonal_features = self.mlp(positonal_features.view(N, C, -1).permute(0, 2, 1))
            positonal_features = positonal_features.permute(0, 2, 1).view(N, C, H, W)
        return positonal_features