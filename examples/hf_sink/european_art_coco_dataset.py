import os
import shutil
import atlas
from datasets import load_dataset
from atlas.tasks.hf.hf import HFDataset
import lance

def main():
    # Define paths
    output_dir = "european_art_coco.lance"
    db_path = "lancedb"
    
    # Clean up previous runs
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    print("Loading dataset...")
    # Load a small, fixed number of rows without streaming for speed
    base_dataset = load_dataset("biglam/european_art_coco_loaded", split="train[:20]", stream=True)
    print("Dataset loaded.")

    # --- Print initial schema ---
    print("\n--- Initial Hugging Face Schema (Pre-Expansion) ---")
    # Must instantiate HFDataset to inspect the schema Atlas will generate
    hf_dataset = HFDataset(base_dataset, expand_level=1)
    print(hf_dataset.schema)
    print("-------------------------------------------------")

    # --- Sink the data ---
    print("\nSinking data...")
    lance_file_path = os.path.join(db_path, output_dir)
    atlas.sink(hf_dataset, lance_file_path)
    print("Sinking complete.")

    # --- Verify the final schema ---
    print("\n--- Final Lance Schema (Post-Expansion) ---")
    output_dataset = lance.dataset(lance_file_path)
    final_schema = output_dataset.schema
    print(final_schema)
    print("-------------------------------------------")

    # --- Programmatic Check for Expansion ---
    print("\n--- Verification ---")
    # Check for fields that should exist after expansion
    expanded_fields = {"objects_bbox", "objects_category_id", "objects_id"}
    final_field_names = set(final_schema.names)
    
    if expanded_fields.issubset(final_field_names):
        print("✅ Verification successful: 'objects' column was expanded.")
    else:
        print("❌ Verification failed: 'objects' column was NOT expanded.")
        missing = expanded_fields - final_field_names
        print(f"   Missing expanded fields: {', '.join(missing)}")
    print("----------------------")
    os._exit(0)


if __name__ == "__main__":
    main()
