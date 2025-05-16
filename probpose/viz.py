from matplotlib import cm
import numpy as np

def overlay_heatmap_on_image(image, heatmap, colormap='jet'):
    """
    Overlay a heatmap on an image.

    Parameters:
        image (numpy.ndarray): The original image.
        heatmap (numpy.ndarray): The heatmap to overlay, shape (K, H, W), already normalized.
        colormap (str): The colormap to use for the heatmap.

    Returns:
        numpy.ndarray: The image with the heatmap overlay.
    """
    
    # Apply the colormap
    colormap = cm.get_cmap(colormap)

    colored_heatmaps = []

    for i in range(heatmap.shape[0]):
        
        # Apply the colormap
        colored_heatmap = colormap(heatmap[i])[:, :, :3]

        # Heatmap values of close to 0 are transparent
        colored_heatmap[heatmap[i] < 0.01] = 0
        colored_heatmaps.append(colored_heatmap)

    colored_heatmaps = np.array(colored_heatmaps)
    combined_heatmap = np.sum(colored_heatmaps, axis=0)
    combined_heatmap = (combined_heatmap * 255).astype(np.uint8)
    combined_heatmap = np.clip(combined_heatmap, 0, 255)
    # Overlay the heatmap on the image
    overlayed_image = np.clip(image + combined_heatmap, 0, 255).astype(np.uint8)

    return overlayed_image