import atlas
import os
import lance
from datasets import load_dataset
import json

# Traditional file-based example
print("--- Running CoT dataset sinking example with a file ---")

# 1. Create a dummy JSONL file
dummy_cot_content = [
    {"question": "What is the capital of France?", "thought": "The user is asking for the capital of France. I know that Paris is the capital of France.", "answer": "Paris"},
    {"question": "Who wrote Hamlet?", "thought": "The user is asking about the author of Hamlet. I know that William Shakespeare wrote Hamlet.", "answer": "William Shakespeare"}
]
jsonl_path = "examples/data/dummy_cot.jsonl"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
with open(jsonl_path, "w") as f:
    for record in dummy_cot_content:
        f.write(json.dumps(record) + "\n")

# 2. Define the path for the Lance dataset
lance_path_file = "examples/data/cot_file.lance"



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
assert table_file.to_pydict()['question'][0] == first_line['question'], "Data mismatch in 'question' column"
assert table_file.to_pydict()['answer'][0] == first_line['answer'], "Data mismatch in 'answer' column"

print("CoT dataset sinking example with a file executed successfully.")


# Hugging Face dataset example
print("--- Running CoT dataset sinking example with a Hugging Face dataset ---")

# 1. Load the CommonsenseQA dataset from Hugging Face
print("Loading CommonsenseQA dataset from Hugging Face...")
dataset = load_dataset("commonsense_qa", split="train")

# 2. Define the path for the Lance dataset
lance_path_hf = "examples/data/cot_hf.lance"
os.makedirs(os.path.dirname(lance_path_hf), exist_ok=True)

# 3. Create a generator to feed the data to the sink
def data_generator():
    for record in dataset:
        question = record["question"]
        choices = record["choices"]["text"]
        answer_key = record["answerKey"]
        answer = choices[ord(answer_key) - ord('A')]
        thought = f"The user is asking: {question}. The options are {', '.join(choices)}. The correct answer is {answer}."
        yield {
            "question": question,
            "thought": thought,
            "answer": answer,
        }

# 4. Sink the dataset to a Lance dataset
print(f"Sinking dataset to {lance_path_hf}...")
atlas.sink(data_generator(), lance_path_hf, mode="overwrite", options={"task": "cot", "format": "cot"})

# 5. Verify that the dataset was created and is not empty
print("Verifying dataset...")
dataset_hf = lance.dataset(lance_path_hf)
assert dataset_hf.count_rows() == len(dataset), "The number of rows in the dataset does not match the number of rows in the Hugging Face dataset"

# 6. Verify the contents of the dataset
table_hf = dataset_hf.to_table(limit=1)
hf_record = next(iter(dataset))
assert table_hf.to_pydict()['question'][0] == hf_record['question'], "Data mismatch in 'question' column"


print("CoT dataset sinking example with a Hugging Face dataset executed successfully.")

