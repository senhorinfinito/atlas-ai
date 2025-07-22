import os
import shutil
import atlas
from datasets import Dataset
import pyarrow as pa
import lance
import pandas as pd
from atlas.tasks.hf import HFDataset

def main():
    # Define paths
    output_dir = "hf_nested_expansion.lance"
    db_path = "lancedb"
    
    # Clean up previous runs
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    # 1. Design a complex dummy dataset
    dummy_data = [
        {
            "id": 1,
            "image_path": "/path/to/image1.jpg",
            "objects": [
                {"id": 101, "category": "cat"},
                {"id": 102, "category": "dog"},
            ],
        },
        {
            "id": 2,
            "image_path": "/path/to/image2.jpg",
            "objects": [
                {"id": 201, "category": "cat"},
            ],
        },
        {
            "id": 3,
            "image_path": "/path/to/image3.jpg",
            "objects": [], # Empty list
        },
        {
            "id": 4,
            "image_path": "/path/to/image4.jpg",
            "objects": None, # Null list
        },
    ]

    try:
        # 3. Create a Hugging Face Dataset from a pandas DataFrame
        df = pd.DataFrame(dummy_data)
        dataset = Dataset.from_pandas(df)

        # --- Print initial schema ---
        print("\n--- Initial Hugging Face Schema (Pre-Expansion) ---")
        # Must instantiate HFDataset to inspect the schema Atlas will generate
        hf_dataset = atlas.tasks.hf.hf.HFDataset(dataset, expand_level=1, handle_nested_nulls=True)
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
        expected_fields = {
            "id", "image_path", 
            "objects_id", "objects_category"
        }
        final_field_names = set(final_schema.names)
        
        if expected_fields.issubset(final_field_names):
            print("✅ Verification successful: All expected columns were created.")
        else:
            print("❌ Verification failed: Not all columns were expanded correctly.")
            missing = expected_fields - final_field_names
            print(f"   Missing expanded fields: {', '.join(missing)}")
        print("----------------------")

    finally:
        # Clean up
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

if __name__ == "__main__":
    main()
