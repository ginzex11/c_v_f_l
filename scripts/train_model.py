# scripts/train_model.py
from ultralytics import YOLO
import pytest

def test_model_initialization():
    """Test YOLOv8 model initialization."""
    model = YOLO("yolov8n.yaml")
    assert model is not None, "Model initialization failed"

def train_yolo_model(data_path, epochs=100, augment=False):
    """Train YOLOv8 model on custom dataset."""
    model = YOLO("yolov8n.yaml")  # Nano model for edge devices
    if augment:
        model.train(data=data_path, epochs=epochs, imgsz=640, batch=16,
                    hsv_h=0.015, hsv_s=0.3, hsv_v=0.2, flipud=0.3, fliplr=0.3, degrees=5.0)
    else:
        model.train(data=data_path, epochs=epochs, imgsz=640, batch=16)
    return model

def evaluate_model(model, val_data):
    """Evaluate model performance."""
    metrics = model.val(data=val_data)
    return metrics.box.map50  # Updated to access mAP50 correctly

if __name__ == "__main__":
    data_yaml = "dataset/data.yaml"
    baseline_model = train_yolo_model(data_yaml, augment=False)
    baseline_mAP = evaluate_model(baseline_model, data_yaml)
    aug_model = train_yolo_model(data_yaml, augment=True)
    aug_mAP = evaluate_model(aug_model, data_yaml)
    print(f"Baseline mAP50: {baseline_mAP}, Augmented mAP50: {aug_mAP}")