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
from probpose.codec import Codec, ProbMap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for ProbPose")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="head",
        choices=["head", "full"],
        help="Type of model to load (default: head)",
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
        default="384,384",
        help="Input size for the model (default: 768,768)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the output heatmaps",
    )
    args = parser.parse_args()
    input_size = tuple(map(int, args.input_size.split(",")))
    if args.model_type == "head":
        # Load the model
        backbone = RadioBackbone(args.backbone)
        head: ProbMapHead = torch.load(args.model, weights_only=False)
        print(head.normalize_layer)
        model = ProbPoseModel(backbone, head).to("cuda")
    elif args.model_type == "full":
        # Load the full model
        model = torch.load(args.model, weights_only=False)
        model = model.to("cuda")
    codec = Codec(ProbMap(input_size, (96, 96), np.array([0.5] * 20)))
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
    print("Output heatmap shape:", heatmaps.shape)
    print("Output heatmap sums:", heatmaps.sum(axis=0).shape)
    print("Output heatmap 1:", heatmaps[0])
    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        print(f"Heatmap {i}: {heatmap}")
        print(f"Heatmap min: {heatmap.min()}, max: {heatmap.max()}, sum: {heatmap.sum()}")
        if args.normalize:
            heatmap = heatmap / heatmap.max()
        heatmap = cm.inferno(heatmap) * 255
        heatmap = heatmap.astype(np.uint8)
        heatmap = PIL.Image.fromarray(heatmap)
        heatmap.save(output_folder / f"heatmap_{i}.png")

    preds = codec.decode(output)
    print("Predictions:", preds[0])
    print("Probabilities:", preds[1])
    print("Visibilities:", preds[2])
    print("OKS:", preds[3])
    print("Errors:", preds[4])

    # Draw predictions on the image
    for i, (keypoints, probabilities) in enumerate(zip(preds[0][0], preds[1][0])):
        for j, (kp, prob) in enumerate(zip(keypoints, probabilities)):
            print(f"Keypoint {j}: {kp}, Probability: {prob}")
            if prob < 0.9:
                continue
            x, y = int(kp[0]), int(kp[1])
            print(f"Keypoint {j}: ({x}, {y}), Probability: {prob:.2f}")
            if 0 <= x < input_size[0] and 0 <= y < input_size[1]:
                PIL.ImageDraw.Draw(image).ellipse(
                    (x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 0)
                )
                PIL.ImageDraw.Draw(image).text(
                    (x + 10, y - 10), f"{j}: {prob:.2f}", fill=(255, 255, 255)
                )
    # Save the output image
    image.save(output_folder / "output_image.png")

    