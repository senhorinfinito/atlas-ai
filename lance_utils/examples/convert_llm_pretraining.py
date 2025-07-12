# examples/convert_llm_pretraining.py

import os
from pathlib import Path
import urllib.request
import atlas

def download_file(url, dest_path):
    """Downloads a file."""
    if dest_path.exists():
        print(f"{dest_path} already exists. Skipping download.")
        return
    
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest_path)

def main():
    # Setup
    data_dir = Path("temp_data/tinyshakespeare")
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text_file = data_dir / "input.txt"
    
    # Download
    download_file(dataset_url, text_file)
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "tinyshakespeare.lance"

    # Use lance_utils to build the dataset
    print("\nBuilding Lance dataset...")
    atlas.build(
        input_path=str(data_dir),
        output_path=str(output_path)
    )
    
    print(f"\nConversion complete!")
    print(f"Dataset saved to {output_path}")
    print(f"To inspect, run: lance_utils inspect -i {output_path}")
    
    # Clean up
    # shutil.rmtree(data_dir)

if __name__ == "__main__":
    main()
