import atlas
import os
from datasets import load_dataset
import lance
import json

# Traditional file-based example
print("--- Running paired text dataset sinking example with a file ---")

# 1. Create a dummy JSONL file
dummy_paired_text_content = [
    {"sentence1": "The cat sat on the mat.", "sentence2": "A feline was resting on the rug.", "label": 1},
    {"sentence1": "I love to eat pizza.", "sentence2": "I dislike eating pizza.", "label": 0}
]
jsonl_path = "examples/data/stsb_train.jsonl"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
with open(jsonl_path, "w") as f:
    for record in dummy_paired_text_content:
        f.write(json.dumps(record) + "\n")

# 2. Define the path for the Lance dataset
lance_path_file = "examples/data/stsb_file.lance"



# 2. Sink the JSONL file to a Lance dataset
print(f"Sinking {jsonl_path} to {lance_path_file}...")
atlas.sink(jsonl_path, lance_path_file, mode="overwrite")

# 3. Verify that the dataset was created and is not empty
print("Verifying dataset...")
dataset_file = lance.dataset(lance_path_file)
assert dataset_file.count_rows() > 0, "The dataset is empty"

# 4. Verify the contents of the dataset
table_file = dataset_file.to_table()
with open(jsonl_path, "r") as f:
    first_line = json.loads(f.readline())
assert table_file.to_pydict()['sentence1'][0] == first_line['sentence1'], "Data mismatch in 'sentence1' column"
assert table_file.to_pydict()['sentence2'][0] == first_line['sentence2'], "Data mismatch in 'sentence2' column"

print("Paired text dataset sinking example with a file executed successfully.\n")


# Hugging Face dataset example
print("--- Running paired text dataset sinking example with a Hugging Face dataset ---")

# 1. Load the GLUE STSB dataset from Hugging Face
print("Loading GLUE STSB dataset from Hugging Face...")
stsb_dataset = load_dataset("glue", "stsb", split='train')

# 2. Define the path for the Lance dataset
lance_path_hf = "examples/data/stsb_hf.lance"
os.makedirs(os.path.dirname(lance_path_hf), exist_ok=True)

# 3. Sink the paired text dataset to a Lance dataset
print(f"Sinking dataset to {lance_path_hf}...")
atlas.sink(stsb_dataset, lance_path_hf, mode="overwrite", options={"task": "paired_text", "format": "paired_text"})

# 4. Verify that the dataset was created and is not empty
lance_dataset_hf = lance.dataset(lance_path_hf)
assert lance_dataset_hf.count_rows() == len(stsb_dataset), "The number of rows in the dataset does not match the number of records in the original dataset"

# 5. Verify the contents of the dataset
table_hf = lance_dataset_hf.to_table(limit=1)
hf_record = next(iter(stsb_dataset))
assert table_hf.to_pydict()['sentence1'][0] == hf_record['sentence1'], "Data mismatch in 'sentence1' column"
assert table_hf.to_pydict()['sentence2'][0] == hf_record['sentence2'], "Data mismatch in 'sentence2' column"

print("Paired text dataset sinking example with a Hugging Face dataset executed successfully.")
