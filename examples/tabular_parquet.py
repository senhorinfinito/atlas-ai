import atlas
import os
import pandas as pd
import lance

print("--- Running tabular parquet sinking example ---")

# Create a temporary directory to store the data
os.makedirs("examples/data", exist_ok=True)

# Create a dummy Parquet file
data = {"col1": [1, 2, 3], "col2": ["A", "B", "C"]}
df = pd.DataFrame(data)
parquet_path = "examples/data/dummy.parquet"
df.to_parquet(parquet_path, index=False)

# Sink the Parquet file to a Lance dataset
lance_path = "examples/data/tabular_parquet.lance"
print(f"Sinking {parquet_path} to {lance_path}...")
atlas.sink(parquet_path, lance_path, mode="overwrite")

# Verify that the dataset was created and is not empty
print("Verifying dataset...")
dataset = lance.dataset(lance_path)
assert dataset.count_rows() == 3, "The number of rows in the dataset does not match the number of rows in the Parquet file"

# Verify the contents of the dataset
table = dataset.to_table()
original_df = pd.read_parquet(parquet_path)
assert table.num_columns == len(original_df.columns), "Column count mismatch"
assert table.to_pydict()['col1'] == original_df['col1'].to_list(), "Data mismatch in 'col1' column"
assert table.to_pydict()['col2'] == original_df['col2'].to_list(), "Data mismatch in 'col2' column"

print("Tabular parquet sinking example executed successfully.")
