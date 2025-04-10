import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from sklearn.utils import resample
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

# === Paths ===
IMAGE_DIR = Path("datasets/yolo_dataset/images/train")
LABEL_DIR = Path("datasets/yolo_dataset/labels/train")
YOLO_WEIGHTS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt"
]

IMAGE_SIZE = (1024, 1024)
SAMPLE_LIMIT = 100

# === Metric helpers ===

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0.0

def hd95(points1, points2):
    if not len(points1) or not len(points2):
        return np.inf
    dists = cdist(points1, points2)
    return max(np.percentile(np.min(dists, axis=1), 95), np.percentile(np.min(dists, axis=0), 95))

def assd(points1, points2):
    if not len(points1) or not len(points2):
        return np.inf
    dists = cdist(points1, points2)
    return 0.5 * (np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0)))

def extract_surface(mask):
    struct = generate_binary_structure(2, 1)
    eroded = binary_erosion(mask, structure=struct)
    surface = binary_dilation(mask, structure=struct) & ~eroded
    points = np.argwhere(surface)
    if len(points) > 10000:
        points = resample(points, n_samples=10000, random_state=42)
    return points

def decode_txt_mask(txt_path, shape):
    h, w = shape
    mask_stack = []
    anatomy_ids = []

    with open(txt_path, 'r') as f:
        for line in f:
            cls, *coords = map(float, line.strip().split())
            coords = np.array(coords).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            coords = coords.astype(np.int32)
            blank = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(blank, [coords], 1)
            mask_stack.append(blank)
            anatomy_ids.append(int(cls // 10) + 1)

    return mask_stack, anatomy_ids

def merge_by_anatomy(stack, anatomy_ids):
    merged = np.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.uint8)
    for i, mask in enumerate(stack):
        a = anatomy_ids[i] - 1
        if a in [0, 1, 2]:
            resized = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
            merged[a] |= resized
    return merged

# === Evaluation Loop ===

print("Evaluating YOLOv8 segmentation on first 100 images...")

image_list = sorted(list(IMAGE_DIR.glob("*.jpg")))[:SAMPLE_LIMIT]

for weight in YOLO_WEIGHTS:
    print(f"\nEvaluating {weight}...")
    model = YOLO(weight)

    metrics = {
        "fracture_iou": 0.0,
        "fracture_hd95": 0.0,
        "fracture_assd": 0.0,
        "anatomical_iou": 0.0,
        "anatomical_hd95": 0.0,
        "anatomical_assd": 0.0
    }
    count_fracture = 0
    count_anatomy = 0

    for img_path in tqdm(image_list, desc="Processing"):
        label_path = LABEL_DIR / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        image = cv2.resize(image, IMAGE_SIZE)
        h, w = IMAGE_SIZE

        # Ground truth
        gt_stack, gt_anatomy = decode_txt_mask(label_path, (h, w))
        gt_anatomy_mask = merge_by_anatomy(gt_stack, gt_anatomy)

        # Inference
        results = model.predict(image, imgsz=1024, conf=0.1, iou=0.1, verbose=False)
        masks = results[0].masks
        boxes = results[0].boxes

        pred_stack = []
        pred_anatomy = []
        if masks is not None:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for i, m in enumerate(masks.data):
                mask = m.cpu().numpy().astype(np.uint8)
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_stack.append(mask)
                pred_anatomy.append((class_ids[i] // 10) + 1)

        pred_anatomy_mask = merge_by_anatomy(pred_stack, pred_anatomy)

        # === Fracture Metrics ===
        for i in range(len(gt_stack)):
            if i >= len(pred_stack):
                continue
            iou_score = iou(gt_stack[i], pred_stack[i])
            metrics["fracture_iou"] += iou_score
            metrics["fracture_hd95"] += hd95(extract_surface(gt_stack[i]), extract_surface(pred_stack[i]))
            metrics["fracture_assd"] += assd(extract_surface(gt_stack[i]), extract_surface(pred_stack[i]))
            count_fracture += 1

        # === Anatomy Metrics ===
        for i in range(3):
            iou_score = iou(gt_anatomy_mask[i], pred_anatomy_mask[i])
            metrics["anatomical_iou"] += iou_score
            metrics["anatomical_hd95"] += hd95(extract_surface(gt_anatomy_mask[i]), extract_surface(pred_anatomy_mask[i]))
            metrics["anatomical_assd"] += assd(extract_surface(gt_anatomy_mask[i]), extract_surface(pred_anatomy_mask[i]))
            count_anatomy += 1

    # === Print final metrics ===
    print(f"  fracture_iou: {metrics['fracture_iou'] / count_fracture:.4f}")
    print(f"  fracture_hd95: {metrics['fracture_hd95'] / count_fracture:.4f}")
    print(f"  fracture_assd: {metrics['fracture_assd'] / count_fracture:.4f}")
    print(f"  anatomical_iou: {metrics['anatomical_iou'] / count_anatomy:.4f}")
    print(f"  anatomical_hd95: {metrics['anatomical_hd95'] / count_anatomy:.4f}")
    print(f"  anatomical_assd: {metrics['anatomical_assd'] / count_anatomy:.4f}")