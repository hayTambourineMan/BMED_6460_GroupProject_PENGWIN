import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import random

# === Paths ===
yolo_weight_path = "yolov8x.pt"  # Change to your trained weights
image_dir = Path("datasets/yolo_dataset/images/train")
save_dir = Path("inference")
save_dir.mkdir(parents=True, exist_ok=True)

# === Generate a consistent colormap for 30 classes ===
def get_color_map(num_classes=30):
    random.seed(42)
    return [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(num_classes)]

color_map = get_color_map()

# === Load YOLO model ===
model = YOLO(yolo_weight_path)

# === Select first 20 images ===
image_files = sorted(list(image_dir.glob("*.jpg")))[:20]

# === Run inference and save overlays ===
for img_path in tqdm(image_files, desc="Running YOLO Inference"):
    # Load grayscale image
    image_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    # Run inference
    results = model.predict(source=str(img_path), save=False, verbose=False)[0]

    if results.masks is not None:
        for i, mask in enumerate(results.masks.data):
            class_id = int(results.boxes.cls[i].item()) if hasattr(results.boxes, 'cls') else 0
            color = color_map[class_id % len(color_map)]
            mask_np = mask.cpu().numpy().astype(np.uint8)

            # Resize if necessary
            if mask_np.shape != image_gray.shape:
                mask_np = cv2.resize(mask_np, (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Overlay mask as filled contour
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_rgb, contours, -1, color, thickness=cv2.FILLED)

    # Save overlayed image
    save_path = save_dir / f"{img_path.stem}_overlay.png"
    cv2.imwrite(str(save_path), image_rgb)

print(f"\nDone! Overlay images saved to: {save_dir.resolve()}")
