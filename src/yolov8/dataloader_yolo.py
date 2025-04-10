# dataloader_yolo.py
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# === Correct import from main utils file ===
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pengwin_utils import load_image, load_masks, visualize_drr

# === CONFIG ===
INPUT_IMG_DIR = Path("../../train/input/images/x-ray")
INPUT_MASK_DIR = Path("../../train/output/images/x-ray")
YOLO_DATASET_DIR = Path("./datasets/yolo_dataset")
IMG_DIR = YOLO_DATASET_DIR / "images"
LBL_DIR = YOLO_DATASET_DIR / "labels"
MAX_IMAGES = 10000
OUTPUT_RESOLUTION = (512, 512)

# Create all split directories
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(IMG_DIR / split, exist_ok=True)
    os.makedirs(LBL_DIR / split, exist_ok=True)

# === SPLIT DATA ===
image_paths = sorted(list(INPUT_IMG_DIR.glob("*.tif")))[:MAX_IMAGES]
train_val_paths, test_paths = train_test_split(image_paths, test_size=0.1, random_state=42)
train_paths, val_paths = train_test_split(train_val_paths, test_size=0.1111, random_state=42)  # 0.1111 × 0.9 ≈ 0.1

split_map = {
    "train": train_paths,
    "val": val_paths,
    "test": test_paths
}

def masks_to_yolo_instances(masks, category_ids, fragment_ids, orig_shape, resized_shape):
    h_orig, w_orig = orig_shape
    h_new, w_new = resized_shape
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig
    yolo_labels = []

    for mask, cat_id, frag_id in zip(masks, category_ids, fragment_ids):
        class_id = (cat_id - 1) * 10 + frag_id
        if class_id == 0:
            continue

        mask = mask.astype(np.uint8)
        mask[mask > 0] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.shape[0] < 6:
                continue
            contour = contour.squeeze()
            if len(contour.shape) != 2 or contour.shape[0] < 6:
                continue

            contour = contour.astype(np.float32)
            contour[:, 0] *= scale_x / w_new
            contour[:, 1] *= scale_y / h_new
            segmentation = contour.flatten().tolist()

            if not all(0.0 <= v <= 1.0 for v in segmentation):
                continue

            segmentation = [round(x, 6) for x in segmentation]
            yolo_labels.append(f"{class_id} " + " ".join(map(str, segmentation)))

    return yolo_labels

# === CONVERT AND SAVE ===
for split, paths in split_map.items():
    print(f"Preparing {len(paths)} images for {split}...")
    for image_path in tqdm(paths, desc=f"Converting {split}"):
        base_name = image_path.stem
        mask_path = INPUT_MASK_DIR / f"{base_name}.tif"
        image = load_image(image_path)
        image = np.squeeze(image)
        original_shape = image.shape[:2]

        masks, cat_ids, frag_ids = load_masks(mask_path)
        yolo_lines = masks_to_yolo_instances(masks, cat_ids, frag_ids, original_shape, OUTPUT_RESOLUTION)

        image = visualize_drr(image)
        image = cv2.resize(image, OUTPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)

        out_img_path = IMG_DIR / split / f"{base_name}.jpg"
        out_lbl_path = LBL_DIR / split / f"{base_name}.txt"

        cv2.imwrite(str(out_img_path), image)
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))

print(f"\nYOLOv8 dataset with train/val/test split saved to: {YOLO_DATASET_DIR.resolve()}")