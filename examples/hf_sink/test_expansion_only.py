import shutil
from datasets import load_dataset, Dataset
import pyarrow as pa

import atlas
import lance
from atlas.tasks.hf.hf import HFDataset

# 1. Load a text-only dataset with nested features
print("Loading dataset...")
dataset = load_dataset("NousResearch/Hermes-3-Dataset", split="train[:10]")
print("Dataset loaded.")

# 2. Instantiate HFDataset with expansion enabled
# The Hermes-3-Dataset has a 'conversations' column which is a list of dicts.
# Per the new logic, this should NOT be expanded.
hf_dataset = HFDataset(dataset)

# 3. Ingest data
output_dir = "text_hf_not_expanded.lance"
print("Sinking data...")
atlas.sink(hf_dataset, output_dir)
print("Sinking complete.")

# 4. Read the data back and verify
retrieved_data = lance.dataset(output_dir)
table = retrieved_data.to_table()

# 5. Verify the schema and data
print("Schema:", table.schema)

# Verify that the 'conversations' column was NOT expanded
schema_names = table.schema.names
assert "conversations" in schema_names
assert not any("conversations_" in name for name in schema_names)

# Verify the type of the 'conversations' column
conversations_field = table.schema.field("conversations")
assert pa.types.is_list(conversations_field.type)
assert pa.types.is_struct(conversations_field.type.value_type)


# Verify row count
assert retrieved_data.count_rows() == 10

print("\nSuccessfully ingested and verified the text-only dataset with non-expanded list of structs.")
print(f"Retrieved {retrieved_data.count_rows()} rows.")
print(table.to_pandas().head(3))

# Clean up
shutil.rmtree(output_dir)