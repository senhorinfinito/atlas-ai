import os
import shutil
import atlas
from atlas.tasks.hf.hf import HFDataset
from datasets import load_dataset

def main():
    # Define paths
    output_dir = "cova_coco_v2.lance"
    db_path = "lancedb"  # Directory for the database
    
    # Clean up previous runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    print("Loading dataset in streaming mode...")
    base_dataset = load_dataset("CreatlV/cova-coco-v2", split="train", streaming=True)
    print("Taking first 100 elements...")
    streamed_dataset = base_dataset.take(100)
    print("Dataset loaded.")

    print("Sinking data...")
    # Sink to a file within the lancedb directory for clarity
    lance_file_path = os.path.join(db_path, output_dir)
    atlas.sink(streamed_dataset, lance_file_path, task="hf", expand_level=1)
    print("Sinking complete.")
    
    # Clean up
    shutil.rmtree(db_path)
    
    # fix the need for this
    os._exit(0)

if __name__ == "__main__":
    main()
