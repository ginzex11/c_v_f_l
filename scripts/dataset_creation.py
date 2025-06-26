# scripts/dataset_creation.py
import os
import cv2
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor
import shutil
import random
import torch
import logging
import psutil
from retrying import retry
import requests
import accelerate
from collections import defaultdict
from PIL import Image

# Suppress external library logs
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler with UTF-8
try:
    file_handler = logging.FileHandler('run.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    logger.error(f"Failed to open run.log: {e}", exc_info=True)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB")

def retry_if_network_error(exception):
    """Retry on network-related exceptions."""
    return isinstance(exception, (requests.exceptions.RequestException, requests.exceptions.ConnectionError))

@retry(stop_max_attempt_number=3, wait_fixed=5000, retry_on_exception=retry_if_network_error)
def load_florence2():
    """Load Florence-2-large model and processor with retry logic."""
    logger.debug("Starting to load Florence-2-large model and processor")
    log_memory_usage()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            device_map="cpu"
        )
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        logger.info("Florence-2-large loaded successfully")
        return model, processor
    except Exception as e:
        logger.error(f"Failed to load Florence-2-large: {e}", exc_info=True)
        raise

def filter_relevant_images(captions_file, image_dir):
    """Filter images, prioritizing pet-related captions."""
    image_extensions = (".jpg", ".jpeg", ".png")
    image_dir = Path(image_dir)
    all_images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not captions_file.exists():
        logger.warning(f"Captions file {captions_file} not found, using all {len(all_images)} images")
        return all_images
    
    try:
        logger.info(f"Reading captions from {captions_file}")
        df = pd.read_csv(captions_file, encoding="utf-8", sep="|", on_bad_lines="skip")
        logger.debug(f"CSV columns: {df.columns.tolist()}")
        logger.debug(f"Sample data (first 5 rows):\n{df.head().to_string()}")
        
        image_col = 'image_name'
        caption_col = ' comment'
        if image_col not in df.columns or caption_col not in df.columns:
            logger.error(f"Required columns '{image_col}' or '{caption_col}' not found in CSV")
            return all_images
        
        pet_keywords = ["dog", "cat", "horse", "puppy", "kitten"]
        person_keywords = ["person", "people", "man", "woman", "child"]
        pet_images = set()
        person_images = set()
        keyword_matches = 0
        for _, row in df.iterrows():
            caption = str(row[caption_col]).lower()
            img_name = str(row[image_col]).strip()
            img_path = image_dir / img_name
            if not img_path.exists():
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = image_dir / (img_name + ext)
                    if img_path.exists():
                        break
            if img_path.exists():
                if any(keyword in caption for keyword in pet_keywords):
                    pet_images.add(img_path)
                    keyword_matches += 1
                elif any(keyword in caption for keyword in person_keywords):
                    person_images.add(img_path)
                    keyword_matches += 1
        relevant_images = list(pet_images) + list(person_images - pet_images)
        logger.info(f"Found {keyword_matches} captions with keywords")
        logger.info(f"Found {len(relevant_images)} relevant images ({len(pet_images)} pet, {len(person_images)} person)")
        return relevant_images if relevant_images else all_images
    except Exception as e:
        logger.error(f"Error reading captions file: {e}", exc_info=True)
        return all_images

def annotate_images(image_dir, captions_file, output_dir, min_obj_size=30, max_obj_size=500, min_img_size=(320, 240), max_img_size=(1280, 720)):
    """Annotate images using Florence-2-large and save in YOLO format."""
    logger.info("Starting image annotation")
    try:
        model, processor = load_florence2()
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    annotations = []
    size_counts = defaultdict(int)
    all_labels = set()
    
    images = filter_relevant_images(captions_file, image_dir)
    logger.info(f"Found {len(images)} relevant images")
    
    # Limit to 5000 images
    images = images[:5000]
    logger.info(f"Processing {len(images)} images")
    
    for i, img_path in enumerate(images):
        logger.debug(f"Processing image {i+1}/{len(images)}: {img_path.name}")
        try:
            img = Image.open(img_path).convert("RGB")
            h, w = img.size[1], img.size[0]
            size_counts[f"{w}x{h}"] += 1
            if not (min_img_size[0] <= w <= max_img_size[0] and min_img_size[1] <= h <= max_img_size[1]):
                logger.info(f"Skipping image {img_path.name}: invalid size ({w}x{h})")
                continue
        
            # Florence-2 inference
            logger.debug(f"Preparing inputs for {img_path.name}")
            task_prompt = "<OD>"
            inputs = processor(text=task_prompt, images=img, return_tensors="pt")
            logger.debug(f"Running inference for {img_path.name}")
            log_memory_usage()
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )
            logger.debug(f"Processing predictions for {img_path.name}")
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            predictions = processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(w, h)
            )
            logger.debug(f"Raw predictions: {predictions}")
            
            bboxes = predictions.get("<OD>", {}).get("bboxes", [])
            labels = predictions.get("<OD>", {}).get("labels", [])
            all_labels.update(labels)
            
            valid_bboxes = []
            for bbox, label in zip(bboxes, labels):
                label = label.lower()
                person_labels = ["person", "human", "people", "man", "woman", "child", "boy", "girl", "human face"]
                pet_labels = ["dog", "cat", "horse", "puppy", "kitten"]
                if label in person_labels:
                    cls = "person"
                elif label in pet_labels:
                    cls = "pet"
                else:
                    continue
                logger.debug(f"Processing bbox for label {label}: {bbox}")
                x, y, x2, y2 = bbox
                width, height = x2 - x, y2 - y
                if min_obj_size <= width <= max_obj_size and min_obj_size <= height <= max_obj_size:
                    valid_bboxes.append({
                        "class": cls,
                        "x": (x + x2) / 2,
                        "y": (y + y2) / 2,
                        "width": width,
                        "height": height
                    })
            
            if valid_bboxes:
                annotations.append({"image": img_path, "bboxes": valid_bboxes})
                save_yolo_annotation(img_path, valid_bboxes, output_dir)
                logger.info(f"Annotated {img_path.name} with {len(valid_bboxes)} valid bounding boxes")
            else:
                logger.info(f"No valid bounding boxes for {img_path.name}")
        
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}", exc_info=True)
            continue
    
    logger.info(f"Completed annotation: {len(annotations)} images annotated")
    logger.info(f"Image size distribution: {dict(size_counts)}")
    logger.debug(f"All detected labels: {sorted(all_labels)}")
    return annotations

def save_yolo_annotation(img_path, bboxes, output_dir):
    """Save bounding boxes in YOLO format."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"Failed to read image {img_path}")
        return
    h, w = img.shape[:2]
    txt_path = output_dir / f"{img_path.stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for box in bboxes:
            cls = 0 if box["class"] == "person" else 1  # person: 0, pet: 1
            x_center, y_center = box["x"] / w, box["y"] / h
            width, height = box["width"] / w, box["height"] / h
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def split_dataset(annotations, output_dir):
    """Split into train (up to 500) and val (up to 100) sets per class."""
    output_dir = Path(output_dir)
    try:
        for cls in ["person", "pet"]:
            cls_annotations = [ann for ann in annotations if cls in [b["class"] for b in ann["bboxes"]]]
            logger.info(f"Found {len(cls_annotations)} images for class {cls}")
            if len(cls_annotations) < 1:
                logger.warning(f"No images found for class {cls}")
                continue
            random.shuffle(cls_annotations)
            train_count = min(len(cls_annotations), 500)
            val_count = min(len(cls_annotations) - train_count, 100)
            train = cls_annotations[:train_count]
            val = cls_annotations[train_count:train_count + val_count]
            for split, name in [(train, "train"), (val, "val")]:
                split_dir = output_dir / name
                split_dir.mkdir(exist_ok=True)
                for ann in split:
                    img_path = ann["image"]
                    shutil.copy(img_path, split_dir / img_path.name)
                    shutil.copy(output_dir / f"{img_path.stem}.txt", split_dir / f"{img_path.stem}.txt")
                logger.info(f"Copied {len(split)} images to {split_dir}")
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}", exc_info=True)

def test_dataset_creation(dataset_dir, output_dir):
    """Test dataset creation process."""
    logger.debug("Starting dataset creation test")
    try:
        captions_file = dataset_dir / "results.csv"
        annotations = annotate_images(dataset_dir, captions_file, output_dir)
        if not annotations:
            logger.warning("No valid annotations generated; checking if images were processed")
        split_dataset(annotations, output_dir)
        logger.info("Dataset creation test passed")
        return True
    except Exception as e:
        logger.error(f"Dataset creation test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.debug("Starting script execution")
    dataset_dir = Path("C:/School/Afeka/computer vision/final project/flickr30k_images")
    output_dir = Path("C:/School/Afeka/computer vision/final project/dataset")
    output_dir.mkdir(exist_ok=True)
    log_memory_usage()
    if test_dataset_creation(dataset_dir, output_dir):
        logger.info("Dataset creation completed")
    else:
        logger.error("Dataset creation failed")
