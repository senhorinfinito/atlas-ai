import os
import shutil
import atlas
from datasets import load_dataset
from atlas.tasks.hf.hf import HFDataset
import lance

def main():
    # Define paths
    output_dir = "european_art_coco.lance"
    db_path = "lancedb"  # Directory for the database
    
    # Clean up previous runs
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    print("Loading dataset in streaming mode...", flush=True)
    base_dataset = load_dataset("biglam/european_art_coco_loaded", split="train", streaming=True)
    streamed_dataset = base_dataset.take(100)
    print("Dataset loaded.", flush=True)

    # --- Print initial schema ---
    print("\n--- Initial Hugging Face Schema ---", flush=True)
    # Must instantiate HFDataset to inspect the schema Atlas will generate
    hf_dataset = HFDataset(streamed_dataset, expand_level=1)
    print(hf_dataset.schema, flush=True)
    print("------------------------------------", flush=True)

    # --- Sink the data ---
    print("\nSinking data...", flush=True)
    lance_file_path = os.path.join(db_path, output_dir)
    # Use the hf_dataset instance directly to avoid re-creating it inside sink
    atlas.sink(hf_dataset, lance_file_path)
    print("Sinking complete.", flush=True)

    # --- Verify the final schema ---
    print("\n--- Final Lance Schema ---", flush=True)
    output_dataset = lance.dataset(lance_file_path)
    final_schema = output_dataset.schema
    print(final_schema, flush=True)
    print("--------------------------", flush=True)

    # --- Programmatic Check for Expansion ---
    print("\n--- Verification ---", flush=True)
    # Check for fields that should exist after expansion
    expanded_fields = {"objects_area", "objects_bbox", "objects_category_id", "objects_image_id", "objects_iscrowd", "objects_segmentation"}
    final_field_names = set(final_schema.names)
    
    if expanded_fields.issubset(final_field_names):
        print("✅ Verification successful: 'objects' column was expanded.", flush=True)
    else:
        print("❌ Verification failed: 'objects' column was NOT expanded.", flush=True)
        missing = expanded_fields - final_field_names
        print(f"   Missing expanded fields: {', '.join(missing)}", flush=True)

    print("----------------------", flush=True)
    
    # Clean up
    shutil.rmtree(db_path)
    
    # Terminate the script forcefully to avoid hanging issues.
    os._exit(0)
    

if __name__ == "__main__":
    main()
    os._exit(0)
    
