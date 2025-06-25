# scripts/inspect_csv.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def inspect_csv(file_path):
    """Inspect CSV file structure and sample data."""
    try:
        logger.info(f"Reading {file_path}")
        df = pd.read_csv(file_path, encoding="utf-8", sep="|", on_bad_lines="skip")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"\nFirst 5 rows:\n{df.head().to_string()}")
        logger.info(f"Total rows: {len(df)}")
        # Check for keywords
        keywords = ["person", "people", "man", "woman", "child", "dog", "cat", "horse", "puppy", "kitten"]
        caption_col = next((col for col in df.columns if any(k in col.lower().strip() for k in ["comment", "caption", "description"]) and "number" not in col.lower()), None)
        if caption_col:
            matches = df[caption_col].str.lower().str.contains('|'.join(keywords), na=False).sum()
            logger.info(f"Rows with keywords in '{caption_col}': {matches}")
        else:
            logger.warning("No caption column found")
    except Exception as e:
        logger.error(f"Error inspecting CSV: {e}", exc_info=True)

if __name__ == "__main__":
    file_path = "C:/School/Afeka/computer vision/final project/flickr30k_images/results.csv"
    inspect_csv(file_path)