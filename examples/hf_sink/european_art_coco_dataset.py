import os
import shutil
import atlas
from datasets import load_dataset

def main():
    # Define paths
    db_path = "lancedb"
    
    # Clean up previous runs
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    try:
        print("Loading one data point for inspection...")
        # Load a single row to inspect its structure
        base_dataset = load_dataset("biglam/european_art_coco_loaded", split="train[:1]")
        print("Dataset loaded.")

        # --- DIAGNOSTIC STEP ---
        print("\n--- Inspecting the 'image' field from a single data point ---")
        for example in base_dataset:
            image_data = example['image']
            print(f"Type of 'image' field: {type(image_data)}")
            print(f"Value of 'image' field (first 100 bytes): {str(image_data)[:100]}...")
            
            # Check if it's a dict with 'bytes'
            if isinstance(image_data, dict):
                print(f"Keys in image dict: {image_data.keys()}")

            # Check if it's a PIL Image by checking for a 'format' attribute
            if hasattr(image_data, 'format'):
                print(f"Object appears to be a PIL Image. Format: {image_data.format}")
            break
        print("-----------------------------------------------------------")

    finally:
        # Clean up
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

if __name__ == "__main__":
    main()
