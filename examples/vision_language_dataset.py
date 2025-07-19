import atlas
import os
import json
from PIL import Image
import numpy as np
import lance

print("--- Running vision-language dataset sinking example with a file ---")

# Create dummy image files
os.makedirs("examples/data/vl_images", exist_ok=True)
image1 = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
image1.save("examples/data/vl_images/image1.png")
image2 = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
image2.save("examples/data/vl_images/image2.png")


# Create a dummy jsonl file
dummy_vl_content = [
    {"image": "examples/data/vl_images/image1.png", "text": "This is the first image."},
    {"image": "examples/data/vl_images/image2.png", "text": "This is the second image."}
]
vl_file_path = "examples/data/dummy_vl.jsonl"
with open(vl_file_path, "w") as f:
    for record in dummy_vl_content:
        f.write(json.dumps(record) + "\n")

# Sink the vision-language dataset to a Lance dataset
lance_path = "examples/data/vision_language.lance"
print(f"Sinking {vl_file_path} to {lance_path}...")
atlas.sink(vl_file_path, lance_path, mode="overwrite")

# Verify that the dataset was created and is not empty
print("Verifying dataset...")
dataset = lance.dataset(lance_path)
assert dataset.count_rows() == 2, "The number of rows in the dataset does not match the number of lines in the jsonl file"

# Verify the contents of the dataset
table = dataset.to_table()
expected_texts = ["This is the first image.", "This is the second image."]
assert table.to_pydict()["text"] == expected_texts, "The text content does not match"

print("Vision-Language dataset sinking example executed successfully.")
