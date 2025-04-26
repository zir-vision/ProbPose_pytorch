from torch import nn, Tensor


class ProbPoseModel(nn.Module):
    def __init__(self, backbone, head):
        super(ProbPoseModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))
