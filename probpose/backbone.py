import torch
from timm.models.vision_transformer import VisionTransformer

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
    
class ScratchViTBackbone(torch.nn.Module):
    def __init__(self, input_image_size: tuple[int, int], patch_size: int):
        super().__init__()
        self.model = VisionTransformer(
            img_size=input_image_size,
            patch_size=patch_size,
            num_classes=0,
            embed_dim=768,
            class_token=False,
            global_pool=''
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, height, width = x.shape
        flat_features = self.model.forward_features(x) # (N, L, C)
        N, L, C = flat_features.shape
        H, W = self.model.patch_embed.dynamic_feat_size((height, width))
        return flat_features.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()