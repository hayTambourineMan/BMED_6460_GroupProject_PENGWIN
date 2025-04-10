import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Set up paths ===
INPUT_IMG_DIR = Path("../train/input/images/x-ray")

# === Import DRR visualization from utils ===
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pengwin_utils import load_image, visualize_drr

# === Get first 10 images ===
image_paths = sorted(list(INPUT_IMG_DIR.glob("*.tif")))[:10]

# === Plotting function ===
def show_drr_comparison(original, enhanced, title):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(enhanced, cmap='gray')
    axs[1].set_title("DRR Enhanced")
    axs[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# === Main Loop ===
for image_path in image_paths:
    base_name = image_path.stem
    image = load_image(image_path)      # Load grayscale image
    image = np.squeeze(image)           # Remove singleton channel
    drr_image = visualize_drr(image)    # Apply enhancement

    show_drr_comparison(image, drr_image, title=base_name)
