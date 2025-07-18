import atlas
from datasets import load_dataset
import lance
import os
import json

# Traditional file-based example
print("--- Running similarity dataset sinking example with a file ---")

# 1. Create a dummy JSONL file
dummy_similarity_content = [
    {"sentence1": "The cat sat on the mat.", "sentence2": "A feline was resting on the rug.", "similarity_score": 0.8},
    {"sentence1": "I love to eat pizza.", "sentence2": "I dislike eating pizza.", "similarity_score": 0.1}
]
jsonl_path = "examples/data/dummy_similarity.jsonl"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
with open(jsonl_path, "w") as f:
    for record in dummy_similarity_content:
        f.write(json.dumps(record) + "\n")

# 2. Define the path for the Lance dataset
lance_path_file = "examples/data/similarity_file.lance"



# 2. Sink the JSONL file to a Lance dataset
print(f"Sinking {jsonl_path} to {lance_path_file}...")
atlas.sink(jsonl_path, lance_path_file, mode="overwrite", options={"task": "similarity", "format": "similarity"})

# 3. Verify that the dataset was created and is not empty
print("Verifying dataset...")
dataset_file = lance.dataset(lance_path_file)
assert dataset_file.count_rows() == 2, "The number of rows in the dataset does not match the number of rows in the JSONL file"

# 4. Verify the contents of the dataset
table_file = dataset_file.to_table()
with open(jsonl_path, "r") as f:
    first_line = json.loads(f.readline())
assert table_file.to_pydict()['sentence1'][0] == first_line['sentence1'], "Data mismatch in 'sentence1' column"
assert table_file.to_pydict()['sentence2'][0] == first_line['sentence2'], "Data mismatch in 'sentence2' column"
import numpy as np
assert np.allclose(table_file.to_pydict()['similarity_score'][0], first_line['similarity_score']), "Data mismatch in 'similarity_score' column"

print("Similarity dataset sinking example with a file executed successfully.\n")


# Hugging Face dataset example
print("--- Running similarity dataset sinking example with a Hugging Face dataset ---")

# 1. Load the STS-B dataset from Hugging Face
print("Loading STS-B dataset from Hugging Face...")
stsb_dataset = load_dataset("stsb_multi_mt", "en", split="train")

# 2. Define the path for the Lance dataset
lance_path_hf = "examples/data/stsb_hf.lance"
os.makedirs(os.path.dirname(lance_path_hf), exist_ok=True)

# 3. Create a generator to feed the data to the sink
def data_generator():
    for record in stsb_dataset:
        yield {
            "sentence1": record["sentence1"],
            "sentence2": record["sentence2"],
            "similarity_score": record["similarity_score"],
        }

# 4. Sink the dataset to a Lance dataset
print(f"Sinking dataset to {lance_path_hf}...")
atlas.sink(data_generator(), lance_path_hf, mode="overwrite", options={"task": "similarity", "format": "similarity"})

# 5. Verify that the dataset was created and is not empty
print("Verifying dataset...")
dataset_hf = lance.dataset(lance_path_hf)
assert dataset_hf.count_rows() == len(stsb_dataset), "The number of rows in the dataset does not match the number of records in the original dataset"

# 6. Verify the contents of the dataset
table_hf = dataset_hf.to_table(limit=1)
hf_record = next(iter(stsb_dataset))
assert table_hf.to_pydict()['sentence1'][0] == hf_record['sentence1'], "Data mismatch in 'sentence1' column"
assert table_hf.to_pydict()['sentence2'][0] == hf_record['sentence2'], "Data mismatch in 'sentence2' column"
assert np.allclose(table_hf.to_pydict()['similarity_score'][0], hf_record['similarity_score']), "Data mismatch in 'similarity_score' column"

print("\nSimilarity dataset example with a Hugging Face dataset executed successfully.")

