import torch


def make_radio_backbone(version: str) -> torch.nn.Module:
    model: torch.nn.Module = torch.hub.load(
        "NVlabs/RADIO", "radio_model", version=version
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model
