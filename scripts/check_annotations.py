# check_annotations.py: Count images with person and pet annotations
import os
from pathlib import Path

def count_annotated_images(dataset_dir, image_dir=None):
    """
    Count images with person (class 0) or pet (class 1) annotations and validate image pairings.
    Args:
        dataset_dir (str): Path to directory with .txt annotation files
        image_dir (str, optional): Path to directory with images for validation
    Returns:
        dict: Counts of images with person, pet, and both annotations
    """
    dataset_path = Path(dataset_dir)
    person_count = 0
    pet_count = 0
    both_count = 0
    processed_files = []

    for txt_file in dataset_path.glob("*.txt"):
        try:
            # Validate corresponding image if image_dir is provided
            if image_dir:
                image_file = Path(image_dir) / f"{txt_file.stem}.jpg"
                if not image_file.exists():
                    print(f"Warning: No image found for {txt_file}")
                    continue

            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                has_person = False
                has_pet = False
                for line in lines:
                    if line.strip():  # Skip empty lines
                        try:
                            class_id = int(line.split()[0])
                            if class_id == 0:
                                has_person = True
                            elif class_id == 1:
                                has_pet = True
                        except (IndexError, ValueError):
                            print(f"Invalid format in {txt_file}: {line.strip()}")
                            break
                else:  # Only count if file is valid
                    if has_person and has_pet:
                        both_count += 1
                    elif has_person:
                        person_count += 1
                    elif has_pet:
                        pet_count += 1
                    processed_files.append(txt_file.name)
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")

    return {
        "person_only": person_count,
        "pet_only": pet_count,
        "both": both_count,
        "total_person": person_count + both_count,
        "total_pet": pet_count + both_count,
        "total_images": person_count + pet_count + both_count,
        "processed_files": len(processed_files),
        "file_list": processed_files
    }

if __name__ == "__main__":
    dataset_dir = "C:/School/Afeka/computer vision/final project/dataset"
    image_dir = "C:/School/Afeka/computer vision/final project/flickr30k_images"
    counts = count_annotated_images(dataset_dir, image_dir)
    print("Annotation Counts:")
    print(f"Images with only person: {counts['person_only']}")
    print(f"Images with only pet: {counts['pet_only']}")
    print(f"Images with both person and pet: {counts['both']}")
    print(f"Total images with person: {counts['total_person']}")
    print(f"Total images with pet: {counts['total_pet']}")
    print(f"Total annotated images: {counts['total_images']}")
    print(f"Total processed .txt files: {counts['processed_files']}")
    # Debug: List first few processed files
    print(f"Sample processed files: {counts['file_list'][:5]}")