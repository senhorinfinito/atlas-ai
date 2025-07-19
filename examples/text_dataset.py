import atlas
import os
import lance

print("--- Running text dataset sinking example with a file ---")

# Create a dummy text file
dummy_text_content = """This is the first line.
This is the second line.
This is the third line.
"""
text_file_path = "examples/data/dummy.txt"
os.makedirs(os.path.dirname(text_file_path), exist_ok=True)
with open(text_file_path, "w") as f:
    f.write(dummy_text_content)

# Sink the text dataset to a Lance dataset
lance_path = "examples/data/text.lance"
print(f"Sinking {text_file_path} to {lance_path}...")
atlas.sink(text_file_path, lance_path, mode="overwrite")

# Verify that the dataset was created and is not empty
print("Verifying dataset...")
dataset = lance.dataset(lance_path)
assert dataset.count_rows() == 3, "The number of rows in the dataset does not match the number of lines in the text file"

# Verify the contents of the dataset
table = dataset.to_table()
expected_text = ["This is the first line.", "This is the second line.", "This is the third line."]
assert table.to_pydict()["text"] == expected_text, "The text content does not match"

print("Text dataset sinking example executed successfully.")
