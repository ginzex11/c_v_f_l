# scripts/inference.py
from ultralytics import YOLO
import os
import random
import glob
import shutil

def infer_image(model_path, image_path, conf=0.1, output_dir="output/predictions"):
    """Run inference on a single image and move results to output_dir."""
    model = YOLO(model_path)  # Load model
    results = model.predict(image_path, save=True, conf=conf, show_labels=True, show_conf=True)  # Predict and save
    # Move result to output_dir
    pred_dir = results[0].save_dir  # e.g., runs/detect/predictX
    pred_image = os.path.join(pred_dir, os.path.basename(image_path))
    os.makedirs(output_dir, exist_ok=True)
    shutil.move(pred_image, os.path.join(output_dir, os.path.basename(image_path)))
    return results

if __name__ == "__main__":
    model_path = "runs/detect/train8/weights/best.pt"
    image_dir = "C:/School/Afeka/computer vision/final project/flickr30k_images"
    output_dir = "output/predictions"
    
    # Get 20 random images (re-run to ensure consolidated outputs)
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    random_images = random.sample(image_paths, min(20, len(image_paths)))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    for image_path in random_images:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        print(f"Processing {image_path}")
        infer_image(model_path, image_path, output_dir=output_dir)