import PIL
import PIL.Image
from probpose.dataset import YOLOPoseDataset
from probpose.codec import Codec, ArgMaxProbMap
from probpose.viz import overlay_heatmap_on_image
import numpy as np
from matplotlib import cm
from pathlib import Path
if __name__ == "__main__":
    # codec = ArgMaxProbMap((768, 768), (192, 192), np.array([0.5] * 20))
    codec = Codec((768, 768), (192, 192), np.array([0.5] * 20))
    ds = YOLOPoseDataset(Path("./data/field-synth-2"), "valid", (768, 768), codec)
    for _ in range(10):
        for i in range(len(ds)):
            img, gt = ds[i]
            print(f"{gt['in_image']=}")


    for idx in range(10):
        img, gt = ds[idx]
        img_arr = (img*255).numpy().astype(np.uint8).transpose(1, 2, 0)
        print(f"Image shape: {img_arr.shape}")
        img: PIL.Image.Image = PIL.Image.fromarray(img_arr)
        img.save("debug_codec/image.png")

        resized_img = img.resize((192, 192))
        overlayed_image = overlay_heatmap_on_image(np.array(resized_img), gt["heatmaps"], colormap="inferno")
        overlayed_image = PIL.Image.fromarray(overlayed_image)
        overlayed_image.save("debug_codec/overlayed_image.png")

        for i, heatmap in enumerate(gt["heatmaps"]):
            print(f"Heatmap min: {heatmap.min()}, max: {heatmap.max()}, sum: {heatmap.sum()}")
            heatmap = cm.inferno(heatmap) * 255
            heatmap = heatmap.astype(np.uint8)
            heatmap = PIL.Image.fromarray(heatmap)
            heatmap.save(f"debug_codec/heatmap_{i}.png")
