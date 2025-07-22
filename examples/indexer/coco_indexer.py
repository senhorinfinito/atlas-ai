import os
import shutil
import atlas
from atlas.index import Indexer
from datasets import load_dataset
from atlas.tasks.hf.hf import HFDataset

def main():
    # Define paths
    output_dir = "coco.lance"
    db_path = "lancedb"  # Directory for the database
    
    # Clean up previous runs
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    #if os.path.exists(db_path):
    #    shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    print("Loading dataset in streaming mode...")
    base_dataset = load_dataset("CreatlV/cova-coco-v2", split="train", streaming=True)
    print("Taking first 100 elements...")
    streamed_dataset = base_dataset.take(512)
    print("Dataset loaded.")

    # Wrap the dataset in HFDataset to generate schema metadata and expand nested columns
    hf_dataset = HFDataset(streamed_dataset, expand_level=1)

    print("Sinking data...")
    # Sink to a file within the lancedb directory for clarity
    lance_file_path = os.path.join(db_path, output_dir)
    atlas.sink(hf_dataset, lance_file_path)
    print("Sinking complete.")

    # --- Initialize the Indexer ---
    # The Indexer will connect to the db_path and create/open the table from the lance_file_path
    idx = Indexer(lance_file_path)

    # --- List existing indexes (should be none) ---
    print("Initial indexes:")
    idx.list_indexes()

    # --- Create a vector index on the 'image' column ---
    print("\nCreating vector index...")
    #idx.create_index("image", "vector")
    idx.create_index(column="pixel_values", index_type="vector")
    print("Vector index created.")

    # --- List indexes again to see the new indexes ---
    print("\nFinal indexes:")
    idx.list_indexes()
    
    # Clean up
    #shutil.rmtree(db_path)
    
    # fix the need for this
    os._exit(0)

if __name__ == "__main__":
    main()