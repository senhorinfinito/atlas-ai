import os
import atlas
from atlas.index import Indexer
from datasets import load_dataset
from lancedb.pydantic import LanceModel, vector
import pyarrow as pa

class CifarSchema(LanceModel):
    img: object
    label: int
    text: str
    vector: vector(128)

def main():
    # --- Create a dummy dataset ---
    if os.path.exists("cifar10.lance"):
        os.system("rm -rf cifar10.lance")
    print("Creating dataset...")
    # Use a non-gated dataset
    dataset = load_dataset("cifar10", split="train")
    # Add dummy text and vector columns for the example
    dataset = dataset.map(lambda example: {'text': f'this is image {example["img"]}', 'vector': [0.1] * 128})
    atlas.sink(dataset, "cifar10.lance")
    print("Dataset created.")

    # --- Initialize the Indexer ---
    idx = Indexer("cifar10.lance", schema=CifarSchema)

    # --- List existing indexes (should be none) ---
    print("Initial indexes:")
    idx.list_indexes()

    # --- Create a vector index on the 'vector' column ---
    print("\nCreating vector index...")
    idx.create_index("vector", "vector")
    print("Vector index created.")

    # --- Create an FTS index on the 'text' column ---
    print("\nCreating FTS index...")
    idx.create_index("text", "fts")
    print("FTS index created.")

    # --- List indexes again to see the new indexes ---
    print("\nFinal indexes:")
    idx.list_indexes()

if __name__ == "__main__":
    main()
