import atlas
import os
import lance
from datasets import load_dataset
import json

# Traditional file-based example
print("--- Running instruction dataset sinking example with a file ---")

# 1. Create a dummy JSONL file
dummy_instruction_content = [
    {"instruction": "Why is the sky blue?", "output": "The sky is blue due to a phenomenon called Rayleigh scattering."},
    {"instruction": "What is the boiling point of water?", "output": "The boiling point of water is 100 degrees Celsius or 212 degrees Fahrenheit at standard atmospheric pressure."}
]
jsonl_path = "examples/data/dummy_instruction.jsonl"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
with open(jsonl_path, "w") as f:
    for record in dummy_instruction_content:
        f.write(json.dumps(record) + "\n")

# 2. Define the path for the Lance dataset
lance_path_file = "examples/data/instruction_file.lance"



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
assert table_file.to_pydict()['instruction'][0] == first_line['instruction'], "Data mismatch in 'instruction' column"
assert table_file.to_pydict()['output'][0] == first_line['output'], "Data mismatch in 'output' column"

print("Instruction dataset sinking example with a file executed successfully.\n")


# Hugging Face dataset example
print("--- Running instruction dataset sinking example with a Hugging Face dataset ---")

# 1. Load the databricks-dolly-15k dataset from Hugging Face
print("Loading databricks-dolly-15k dataset from Hugging Face...")
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# 2. Define the path for the Lance dataset
lance_path_hf = "examples/data/instruction_hf.lance"
os.makedirs(os.path.dirname(lance_path_hf), exist_ok=True)

# 3. Sink the dataset to a Lance dataset
print(f"Sinking dataset to {lance_path_hf}...")
atlas.sink(dataset, lance_path_hf, mode="overwrite", options={"task": "instruction", "format": "instruction"})

# 4. Verify that the dataset was created and is not empty
print("Verifying dataset...")
lance_dataset_hf = lance.dataset(lance_path_hf)
assert lance_dataset_hf.count_rows() == len(dataset), "The number of rows in the dataset does not match the number of rows in the Hugging Face dataset"

# 5. Verify the contents of the dataset
table_hf = lance_dataset_hf.to_table(limit=1)
hf_record = next(iter(dataset))
assert table_hf.to_pydict()['instruction'][0] == hf_record['instruction'], "Data mismatch in 'instruction' column"
assert table_hf.to_pydict()['response'][0] == hf_record['response'], "Data mismatch in 'response' column"


print("\nInstruction dataset sinking example with a Hugging Face dataset executed successfully.")
