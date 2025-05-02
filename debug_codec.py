import PIL
from probpose.dataset import YOLOPoseDataset
from probpose.codec import Codec
import numpy as np
from matplotlib import cm
from pathlib import Path
if __name__ == "__main__":
    codec = Codec((768, 768), (192, 192), np.array([0.5] * 20))
    ds = YOLOPoseDataset(Path("./data/field-synth-2"), "train", (768, 768), codec)
    for idx in range(10):
        img, gt = ds[idx]
        img_arr = (img*255).numpy().astype(np.uint8).transpose(1, 2, 0)
        print(f"Image shape: {img_arr.shape}")
        img = PIL.Image.fromarray(img_arr)
        img.save("debug_codec/image.png")
        for i, heatmap in enumerate(gt["heatmaps"]):
            print(f"Heatmap min: {heatmap.min()}, max: {heatmap.max()}")
            heatmap = cm.inferno(heatmap) * 255
            heatmap = heatmap.astype(np.uint8)
            heatmap = PIL.Image.fromarray(heatmap)
            heatmap.save(f"debug_codec/heatmap_{i}.png")
