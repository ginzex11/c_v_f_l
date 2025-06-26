# scripts/run_split.py
from dataset_creation import split_dataset
from pathlib import Path

def run_split():
    """Run dataset split for train/val, using images from flickr30k_images."""
    dataset_dir = Path("C:/School/Afeka/computer vision/final project/flickr30k_images")
    output_dir = Path("C:/School/Afeka/computer vision/final project/dataset")
    # Collect annotations from .txt files
    annotations = []
    for txt in output_dir.glob("*.txt"):
        img_path = dataset_dir / f"{txt.stem}.jpg"
        if not img_path.exists():
            print(f"Warning: No image for {txt}")
            continue
        with open(txt, "r") as f:
            bboxes = []
            for line in f:
                try:
                    cls = int(line.split()[0])
                    bboxes.append({"class": "person" if cls == 0 else "pet"})
                except (ValueError, IndexError):
                    print(f"Invalid format in {txt}")
                    break
            else:  # Only append if valid
                annotations.append({"image": img_path, "bboxes": bboxes})
    print(f"Found {len(annotations)} valid annotations")
    split_dataset(annotations, output_dir)
    print("Dataset split completed")

if __name__ == "__main__":
    run_split()