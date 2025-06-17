"""
Convert YOLO Pose format annotations to COCO format.
"""

from pathlib import Path

import numpy as np
import PIL
import PIL.Image
import pymage_size
import torch
from rich.progress import track
from torch.utils.data import Dataset
from torchvision.transforms import v2

from probpose.codec import Codec
from probpose.util import ProbPoseGroundTruth


def parse_annotations(split_folder: Path, target_single_class: int | None = None):
    annotations = []
    image_paths = list((split_folder / "images").iterdir())
    for i, image_path in enumerate(track(image_paths)):
        width, height = pymage_size.get_image_size(str(image_path)).get_dimensions()
        label_path = split_folder / "labels" / image_path.with_suffix(".txt").name
        if not label_path.exists():
            print(
                f"Label file {label_path} does not exist, skipping image {image_path.name}"
            )
            continue
        with open(label_path, "r") as f:
            lines = f.readlines()
        for ann_num, line in enumerate(lines):
            parts = line.strip().split()
            cls = int(parts[0])
            if target_single_class is not None and cls != target_single_class:
                continue
            cls = 0
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height
            kps = []
            for j in range(5, len(parts), 3):
                visibility = int(parts[j + 2])
                if visibility == 1:
                    visibility = 2
                kps.append(
                    [
                        float(parts[j]) * width,
                        float(parts[j + 1]) * height,
                        visibility,
                    ]
                )
            annotations.append(
                {
                    "image_path": str(image_path),
                    "category_id": cls,
                    "bbox": [
                        x_center - box_width / 2,
                        y_center - box_height / 2,
                        box_width,
                        box_height,
                    ],
                    "keypoints": kps,
                }
            )
    return annotations


def scale_box(image: PIL.Image.Image, bbox, image_size: tuple[int, int], kps: np.ndarray):
    """
    Scale the bounding box and keypoints to the exact target size.
    """
    cropped = image.crop(
        (
            bbox[0],
            bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3],
        )
    )
    scaled = cropped.resize(
        image_size,
        resample=PIL.Image.LANCZOS,
    )
    # Scale the keypoints to the heatmap size
    kps[:, 0] = (kps[:, 0] - bbox[0]) / bbox[2] * image_size[0]
    kps[:, 1] = (kps[:, 1] - bbox[1]) / bbox[3] * image_size[1]
    return scaled, kps


class YOLOPoseDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        codec: Codec,
        target_single_class: int | None = None,
    ):
        self.root = root
        self.split = split
        self.codec = codec
        self.target_single_class = target_single_class
        self.annotations = parse_annotations(root / split, target_single_class)
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> tuple[torch.Tensor, ProbPoseGroundTruth]:
        img = PIL.Image.open(self.annotations[idx]["image_path"]).convert("RGB")
        bbox = self.annotations[idx]["bbox"]
        kps = self.annotations[idx]["keypoints"]

        img, kps = scale_box(img, bbox, self.codec.probmap.input_size, np.array(kps, dtype=np.float32))
        img = self.transform(img)
        kps = kps[None, :, :]
        kps_visible = kps[:, :, 2] == 2
        kps_visibility = np.minimum(kps[:, :, 2], 1)
        kps = kps[:, :, :2]

        encoded = self.codec.encode(kps, kps_visible)

        return img, dict(
            heatmaps=encoded["heatmaps"],
            in_image=encoded["in_image"],
            keypoints_visible=kps_visible,
            keypoints_visibility=kps_visibility,
        )
