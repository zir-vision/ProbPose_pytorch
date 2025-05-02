import argparse
from pathlib import Path

import PIL
import numpy as np
import torch
from torchvision.transforms import v2
from matplotlib import cm

from probpose.model import ProbPoseModel
from probpose.backbone import RadioBackbone
from probpose.head import ProbMapHead
from probpose.codec import Codec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for ProbPose")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image for inference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the output image with predictions",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="radio_v2.5-b",
        help="Backbone model to use (default: radio_v2.5-b)",
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default="768,768",
        help="Input size for the model (default: 768,768)",
    )
    args = parser.parse_args()
    input_size = tuple(map(int, args.input_size.split(",")))
    # Load the model
    backbone = RadioBackbone(args.backbone)
    head: ProbMapHead = torch.load(args.model, weights_only=False)
    print(head.final_layer.weight)
    model = ProbPoseModel(backbone, head).to("cuda")
    codec = Codec(input_size, (192, 192), np.array([0.5] * 20))
    # Load and preprocess the image
    image = PIL.Image.open(args.image).convert("RGB")
    image = image.resize(input_size, PIL.Image.LANCZOS)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    image_tensor = transform(image).to("cuda").unsqueeze(0)
    print("Input image shape:", image_tensor.shape)
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    output_folder = args.output
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    heatmaps = output[0][0].cpu().numpy()
    for i, heatmap in enumerate(heatmaps):
        print(f"Heatmap min: {heatmap.min()}, max: {heatmap.max()}")
        heatmap = cm.inferno(heatmap) * 255
        heatmap = heatmap.astype(np.uint8)
        heatmap = PIL.Image.fromarray(heatmap)
        heatmap.save(output_folder / f"heatmap_{i}.png")

    preds = codec.decode(output)[0]
    print("Predictions:", preds)

    