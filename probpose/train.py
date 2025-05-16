from pathlib import Path
from typing import cast
import sys
from rich.progress import track
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import MLP

from probpose.backbone import RadioBackbone, ScratchViTBackbone
from probpose.codec import Codec, ArgMaxProbMap
from probpose.dataset import YOLOPoseDataset
from probpose.head import ProbMapHead
from probpose.loss import ProbPoseLoss
from probpose.model import ProbPoseModel
from probpose.util import ProbPoseGroundTruth

IMG_SIZE = (384, 384)
EPOCHS = 200
VAL_EVERY = 50
SAVE_EVERY = 50
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
SAVE_FULL = True

LOSS_WEIGHTS = {
    "kpt": 1.0,
    "probability": 1.0,
    "visibility": 0.0,
    "oks": 1.0,
    "error": 1.0,
}

if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir)
    # backbone = RadioBackbone("radio_v2.5-b")
    backbone = ScratchViTBackbone(
        IMG_SIZE,
        16
    )
    head = ProbMapHead(768, 20, [(4, 4), (2, 2), (2, 2)], (512, 256), (4, 4), final_layer_kernel_size=1, freeze_error=True, normalize=1.0)
    model = ProbPoseModel(backbone, head).to("cuda")
    codec = Codec(IMG_SIZE, (96, 96), np.array([0.1] * 20))
    fast_codec = ArgMaxProbMap(IMG_SIZE, (96, 96), np.array([0.1] * 20))
    loss_fn = ProbPoseLoss(fast_codec, freeze_error=True)
    train_ds = YOLOPoseDataset(
        Path("./data/field-synth-2"),
        "train",
        IMG_SIZE,
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
        IMG_SIZE,
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
        weight_decay=0.1
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    
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
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient norm for {name}: {param.grad.norm().item()}")
            # exit()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                    val_losses: list[dict[str, torch.Tensor]] = []
                    val_accuracies: list[dict[str, torch.Tensor]] = []
                    val_max_heatmap = 0.0
                    for img, gt in val_loader:
                        gt = cast(ProbPoseGroundTruth, gt)
                        img = cast(torch.Tensor, img).to("cuda")

                        pred = model(img)
                        losses, acc = loss_fn(gt, pred, compute_acc=True)
                        val_losses.append(losses)
                        val_accuracies.append(acc)
                        max_heatmap = pred[0].max()
                        if max_heatmap > val_max_heatmap:
                            val_max_heatmap = max_heatmap
                        
                    writer.add_scalar("validation/val_max_heatmap", val_max_heatmap.item(), step)
                    writer.add_scalar("validation/val_mean_prob", torch.mean(pred[1]).item(), step)
                    val_loss = torch.sum(
                        torch.stack(
                            [
                                torch.sum(torch.stack([l[k] for l in val_losses])) / len(val_losses)
                                for k in LOSS_WEIGHTS.keys()
                            ]
                        )
                    )

                    writer.add_scalar("validation/loss", val_loss.item(), step)

                    for k in LOSS_WEIGHTS.keys():
                        writer.add_scalar(
                            f"validation/loss/{k}",
                            torch.sum(torch.stack([l[k] for l in val_losses])) / len(val_losses),
                            step,
                        )
                        writer.add_scalar(
                            f"validation/acc/{k}",
                            torch.sum(torch.stack([a[k] for a in val_accuracies])) / len(val_accuracies),
                            step,
                        )

            if i % SAVE_EVERY == 0:
                torch.save(
                    head,
                    out_dir / f"head_epoch_{epoch}_step_{i}.pth",
                )
                if SAVE_FULL:
                    torch.save(
                        model,
                        out_dir / f"model_epoch_{epoch}_step_{i}.pth",
                    )
