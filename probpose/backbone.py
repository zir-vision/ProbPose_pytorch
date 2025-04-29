import torch


class RadioBackbone(torch.nn.Module):
    def __init__(self, version: str):
        super().__init__()
        self.model = torch.hub.load(
            "NVlabs/RADIO", "radio_model", version=version
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary, positonal_features = self.model(x, feature_fmt="NCHW")
        return positonal_features