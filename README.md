# S25 BMED6460 PENGWIN Project

This repository contains the training, evaluation, and inference code for our final project in **BMED 6460**, focused on the **[PENGWIN Challenge 2024](https://pengwin.grand-challenge.org/)**. The challenge involves **pelvic fracture fragment segmentation** in synthetic 2D X-ray images.

---

## Challenge Overview

The [PENGWIN Challenge](https://pengwin.grand-challenge.org/) provides a benchmark for automated segmentation of fracture fragments on pelvic X-rays. The data is generated synthetically from 3D CT scans and includes detailed multi-label annotations across 30 fragment categories.

We focus on **Task 2: 2D Pelvic Fracture Segmentation**, which requires **instance-level segmentation** of both anatomical bones (sacrum, ilium) and individual fracture fragments.

---

## Models Used

This project explores and compares three deep learning models for 2D X-ray instance segmentation:

| Model | Purpose | GitHub Repository |
|-------|---------|-------------------|
| **[YOLOv8](https://github.com/ultralytics/ultralytics)** | Real-time instance segmentation, adapted for 30 fragment labels |
| **[nnUNetv2](https://github.com/MIC-DKFZ/nnUNet)** | Medical segmentation baseline, trained on 2D PNG masks |
| **[SAM2 (Segment Anything v2)](https://github.com/facebookresearch/sam)** | Foundation model fine-tuned with COCO-style masks for fragment detection |

Each model has separate scripts for training, evaluation (IoU, HD95, ASSD), and visualization.

---

## Project Structure

```
.
├── YOLO/
│   ├── train_yolo.ipynb            # Train YOLOv8 for instance segmentation
│   ├── evaluate_yolo.py            # Quantitative benchmark (first 100 images)
│   ├── inference_yolo.py            # Mask overlay + save visual results
│
├── nnUNet/
│   ├── nnunetv2.ipynb            # nnUNetv2 training on fragment masks
│
├── SAM/
│   ├── sam2.ipynb            # SAM2 fine-tuning notebook (Colab-ready)
│
├── utils/
│   ├── evaluation_metrics.py          # Custom metrics functions
│   └── pengwin_utils.py            # Mask decoding + formatting helpers
```

---

## Environment Setup

### YOLOv8 (Ultralytics)
```bash
conda create -n yolov8_env python=3.10 -y
conda activate yolov8_env
pip install ultralytics opencv-python tqdm
```

### nnUNetv2
```bash
conda create -n nnunet_env python=3.10 -y
conda activate nnunet_env
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
pip install SimpleITK scikit-learn
```

Set environment variables:
```bash
export nnUNet_raw="/content/nnUNet_raw"
export nnUNet_preprocessed="/content/nnUNet_preprocessed"
export nnUNet_results="/content/nnUNet_results"
```

### SAM2 (Segment Anything v2)
```bash
git clone https://github.com/facebookresearch/sam
cd sam
pip install -r requirements.txt
pip install -e .
pip install pycocotools supervision
```

---

## Evaluation Metrics

We evaluate predictions using the following segmentation metrics:

- **Fracture-Level Metrics** (per fragment):
  - IoU (Intersection over Union)
  - HD95 (95th percentile Hausdorff Distance)
  - ASSD (Average Symmetric Surface Distance)

- **Anatomical Bone Metrics** (SA, LI, RI):
  - IoU, HD95, and ASSD averaged per anatomical class

Results are calculated on the **first 100 validation images**.

---

## Notes

- The YOLO model was adapted to handle polygon instance masks.
- SAM2 uses fine-tuned checkpoints and COCO-style JSON masks.
- nnUNet was trained on PNG masks and evaluated using uint32 label maps.
- Visualizations and logs are saved during each model’s inference.

---

## Team

- **Henry Liu**
- **Luci Rizor**
- **Mallika Subash**
- **Jesse Yebouet**
