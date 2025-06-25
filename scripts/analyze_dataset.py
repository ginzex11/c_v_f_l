# scripts/analyze_dataset.py
import os
import pandas as pd
from pathlib import Path

def analyze_flickr30k(dataset_dir):
    """Analyze the contents of the Flickr30k dataset."""
    dataset_dir = Path(dataset_dir)
    image_dir = dataset_dir  # Images directly in flickr30k_images
    
    # Count images
    image_extensions = (".jpg", ".jpeg", ".png")
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
    print(f"Total images: {len(images)}")
    
    # Sample first few images
    print("\nSample image files:")
    for img in images[:5]:
        print(f"- {img.name}")
    
    # Check for captions file
    captions_file = dataset_dir / "results.csv"
    if captions_file.exists():
        try:
            df = pd.read_csv(captions_file, encoding="utf-8", sep="|", on_bad_lines="skip")
            print(f"\nCaptions file found: {captions_file}")
            print(f"Total captions: {len(df)}")
            print("\nSample captions (first 5 rows):")
            print(df.head())
        except Exception as e:
            print(f"\nError reading captions file: {e}")
            df = None
    else:
        print(f"\nCaptions file not found at {captions_file}")
    
    return images, captions_file if captions_file.exists() else None

def test_analyze_dataset(dataset_dir):
    """Test dataset analysis function."""
    try:
        images, captions_file = analyze_flickr30k(dataset_dir)
        assert len(images) > 0, "No images found in dataset"
        print("Dataset analysis test passed")
        return True
    except Exception as e:
        print(f"Dataset analysis test failed: {e}")
        return False

if __name__ == "__main__":
    dataset_dir = "C:/School/Afeka/computer vision/final project/flickr30k_images"
    if test_analyze_dataset(dataset_dir):
        print("Dataset analysis complete")
    else:
        print("Dataset analysis failed")
