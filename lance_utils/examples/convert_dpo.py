# examples/convert_dpo.py

import os
import json
from pathlib import Path
import atlas

def create_synthetic_dpo_data(data_path: Path):
    """Creates a synthetic JSONL file for DPO."""
    if data_path.exists():
        print(f"{data_path} already exists. Skipping creation.")
        return
        
    print("Creating synthetic DPO data...")
    dpo_data = [
        {"prompt": "What is the best way to learn Python?", "chosen": "Build projects.", "rejected": "Just read books."},
        {"prompt": "How can I stay motivated?", "chosen": "Set small, achievable goals.", "rejected": "Wait for inspiration to strike."},
        {"prompt": "What's a good breakfast?", "chosen": "Oatmeal and fruit.", "rejected": "A candy bar."}
    ]
    
    with open(data_path, "w") as f:
        for item in dpo_data:
            f.write(json.dumps(item) + "\n")

def main():
    # Setup
    data_dir = Path("temp_data/dpo")
    data_dir.mkdir(parents=True, exist_ok=True)
    dpo_file = data_dir / "dpo_data.jsonl"
    
    # Create synthetic data
    create_synthetic_dpo_data(dpo_file)
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "synthetic_dpo.lance"

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
