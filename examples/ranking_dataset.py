import atlas
import os
from datasets import load_dataset
import lance
import json

# Traditional file-based example
print("--- Running ranking dataset sinking example with a file ---")

# 1. Create a dummy JSONL file
dummy_ranking_content = [
    {"query": "What is the capital of the United States?", "documents": ["Washington D.C.", "New York City", "Los Angeles"]},
    {"query": "What is the highest mountain in the world?", "documents": ["Mount Everest", "K2", "Kangchenjunga"]}
]
jsonl_path = "examples/data/dummy_ranking.jsonl"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
with open(jsonl_path, "w") as f:
    for record in dummy_ranking_content:
        f.write(json.dumps(record) + "\n")

# 2. Define the path for the Lance dataset
lance_path_file = "examples/data/ranking_file.lance"



# 2. Sink the JSONL file to a Lance dataset
print(f"Sinking {jsonl_path} to {lance_path_file}...")
atlas.sink(jsonl_path, lance_path_file, mode="overwrite")

# 3. Verify that the dataset was created and is not empty
print("Verifying dataset...")
dataset_file = lance.dataset(lance_path_file)
assert dataset_file.count_rows() == 2, "The number of rows in the dataset does not match the number of rows in the JSONL file"

# 4. Verify the contents of the dataset
table_file = dataset_file.to_table()
with open(jsonl_path, "r") as f:
    first_line = json.loads(f.readline())
assert table_file.to_pydict()['query'][0] == first_line['query'], "Data mismatch in 'query' column"
assert table_file.to_pydict()['documents'][0] == first_line['documents'], "Data mismatch in 'documents' column"

print("Ranking dataset sinking example with a file executed successfully.\n")


# Hugging Face dataset example
print("--- Running ranking dataset sinking example with a Hugging Face dataset ---")

# 1. Load the MS MARCO dataset from Hugging Face
print("Loading MS MARCO dataset from Hugging Face...")
dataset = load_dataset("ms_marco", "v1.1", split="train")

# 2. Define the path for the Lance dataset
lance_path_hf = "examples/data/ranking_hf.lance"
os.makedirs(os.path.dirname(lance_path_hf), exist_ok=True)

# 3. Sink the ranking dataset to a Lance dataset
print(f"Sinking dataset to {lance_path_hf}...")
atlas.sink(dataset, lance_path_hf, mode="overwrite", task="ranking")

# 4. Verify that the dataset was created and is not empty
lance_dataset_hf = lance.dataset(lance_path_hf)
assert lance_dataset_hf.count_rows() == len(dataset), "The number of rows in the dataset does not match the number of records in the original dataset"

# 5. Verify the contents of the dataset
table_hf = lance_dataset_hf.to_table(limit=1)
hf_record = next(iter(dataset))
assert table_hf.to_pydict()['query'][0] == hf_record['query'], "Data mismatch in 'query' column"

print("Ranking dataset sinking example with a Hugging Face dataset executed successfully.")
