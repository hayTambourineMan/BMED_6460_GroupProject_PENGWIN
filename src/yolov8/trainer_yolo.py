import os
from ultralytics import YOLO

# === CONFIGURATION ===
YOLO_MODELS = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"]
DATA_YAML = "dataset.yaml"
EPOCHS = 100
IMG_SIZE = 512
BATCH_SIZE = 8
PROJECT_NAME = "pengwin_yolo_results"

# === PATHS ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML_PATH = os.path.join(ROOT_DIR, DATA_YAML)

# === TRAIN + EVAL ===
def train_all_yolo_models():
    for model_type in YOLO_MODELS:
        model_name = model_type.replace(".pt", "")
        experiment_name = f"{model_name}_seg_{IMG_SIZE}"

        print(f"\nStarting training for: {model_type}")
        model = YOLO(model_type)

        model.train(
            data=DATA_YAML_PATH,
            imgsz=IMG_SIZE,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=experiment_name,
            task="segment",
            verbose=True,
            workers=4,
            cache=True
        )

        print(f"Training complete for {model_type}. Model saved to: runs/segment/{experiment_name}")

        print(f"Running test evaluation for: {model_type}")
        results = model.val(
            data=DATA_YAML_PATH,
            split="test",
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=experiment_name + "_test",
            task="segment"
        )

        print(f"Test evaluation complete. Results saved to: runs/segment/{experiment_name}_test")

if __name__ == "__main__":
    train_all_yolo_models()