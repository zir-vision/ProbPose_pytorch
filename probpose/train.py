from pathlib import Path
from typing import cast
import sys
from rich.progress import track
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import MLP

from probpose.backbone import RadioBackbone
from probpose.codec import Codec
from probpose.dataset import YOLOPoseDataset
from probpose.head import ProbMapHead
from probpose.loss import ProbPoseLoss
from probpose.model import ProbPoseModel
from probpose.util import ProbPoseGroundTruth

EPOCHS = 10
VAL_EVERY = 50
SAVE_EVERY = 50
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32

LOSS_WEIGHTS = {
    "kpt": 1.0,
    "probability": 1.0,
    "visibility": 1.0,
    "oks": 1.0,
    "error": 1.0,
}

if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir)
    backbone = RadioBackbone("radio_v2.5-b")
    head = ProbMapHead(768, 20, [(4, 4), (2, 2), (2, 2), (2, 2)], (256, 256), (4, 4))
    model = ProbPoseModel(backbone, head).to("cuda")
    codec = Codec((768, 768), (192, 192), np.array([0.1] * 20))
    loss_fn = ProbPoseLoss(codec)
    train_ds = YOLOPoseDataset(
        Path("./data/field-synth-2"),
        "train",
        (768, 768),
        codec,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_ds = YOLOPoseDataset(
        Path("./data/field-synth-2"),
        "valid",
        (768, 768),
        codec,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # img, gt = ds[0]

    # img = img.unsqueeze(0)

    # _, spacial_features = backbone(img, feature_fmt="NCHW")

    # pred = head(spacial_features)

    # loss = loss_fn([gt], pred, torch.ones(1, 20))

    # print(loss)

    for epoch in range(EPOCHS):
        model.train()
        for i, (img, gt) in enumerate(track(train_loader)):
            step = epoch * len(train_loader) + i
            gt = cast(ProbPoseGroundTruth, gt)
            img = cast(torch.Tensor, img).to("cuda")

            optimizer.zero_grad()

            pred = model(img)
            losses: dict = loss_fn(gt, pred)
            loss = torch.sum(
                torch.stack(
                    [
                        losses[k] * LOSS_WEIGHTS[k]
                        for k in LOSS_WEIGHTS.keys()
                    ]
                )
            )
            writer.add_scalar("training/loss", loss.item(), step)
            for k in losses.keys():
                writer.add_scalar(f"training/loss/{k}", losses[k].item(), step)
            writer.add_scalar("training/lr", optimizer.param_groups[0]["lr"], step)
            loss.backward()
            

            optimizer.step()
            scheduler.step()

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
                for k in losses.keys():
                    print(f"{k}: {losses[k].item():.4f}")
            if i % VAL_EVERY == 0:
                 # Validation step
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for img, gt in val_loader:
                        gt = cast(ProbPoseGroundTruth, gt)
                        img = cast(torch.Tensor, img).to("cuda")

                        pred = model(img)
                        losses: dict = loss_fn(gt, pred)
                        loss = torch.sum(
                            torch.stack(
                                [
                                    losses[k] * LOSS_WEIGHTS[k]
                                    for k in LOSS_WEIGHTS.keys()
                                ]
                            )
                        )
                        val_loss += loss.item()
                    val_loss /= len(val_loader)
                    writer.add_scalar("validation/loss", val_loss, step)
                    for k in losses.keys():
                        writer.add_scalar(f"validation/loss/{k}", losses[k].item(), step)
                    print(f"Validation Loss: {val_loss}")
            if i % SAVE_EVERY == 0:
                torch.save(
                    head,
                    out_dir / f"head_epoch_{epoch}_step_{i}.pth",
                )
