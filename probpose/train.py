from pathlib import Path

import numpy as np
import torch

from probpose.backbone import make_radio_backbone
from probpose.codec import Codec
from probpose.dataset import YOLOPoseDataset
from probpose.head import ProbMapHead
from probpose.loss import ProbPoseLoss

if __name__ == "__main__":
    backbone = make_radio_backbone("radio_v2.5-b")
    head = ProbMapHead(768, 20, [(4, 4), (2, 2), (2, 2), (2, 2)], (256, 256), (4, 4))
    codec = Codec((768, 768), (192, 192), np.array([0.1] * 20))
    loss_fn = ProbPoseLoss(codec)
    ds = YOLOPoseDataset(
        Path("./field-synth-2"),
        "valid",
        (768, 768),
        codec,
    )
    img, gt = ds[0]

    img = img.unsqueeze(0)

    _, spacial_features = backbone(img, feature_fmt="NCHW")

    pred = head(spacial_features)

    loss = loss_fn([gt], pred, torch.ones(1, 20))

    print(loss)
