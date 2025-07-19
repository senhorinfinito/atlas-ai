# Atlas: A data-centric AI framework
#
# Copyright (c) 2024-present, Atlas Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
from datasets import load_dataset
import pyarrow as pa

import atlas
from atlas.tasks.hf.hf import HFDataset

# 1. Load a real, public dataset with images in streaming mode
print("Loading dataset in streaming mode...")
base_dataset = load_dataset("ariG23498/coco2017", split="train", streaming=True)
print("Taking first 10 elements...")
streamed_dataset = base_dataset.take(10)
print("Dataset loaded.")

# 2. Instantiate HFDataset with expansion enabled
hf_dataset = HFDataset(streamed_dataset, expand_level=1)

# 3. Ingest data
output_dir = "coco_hf_expanded.lance"
print("Sinking data...")
atlas.sink(hf_dataset, output_dir)
print("Sinking complete.")

# 4. Read the data back
sink = atlas.data_sinks.LanceDataSink(output_dir)
retrieved_data = sink.read()
table = retrieved_data.to_table()

# 5. Verify the schema and data
print("Schema:", table.schema)

# Verify that the 'image' column was ingested as binary
assert pa.types.is_large_binary(table.schema.field("image").type)

# Verify that the nested 'objects' Sequence was expanded into separate columns
# The exact names can vary based on the HF dataset, so we check for substrings
schema_names = table.schema.names

assert any("bbox" in name for name in schema_names)
assert any("categories" in name for name in schema_names)


# Verify row count
print("total rows ", retrieved_data.count_rows())
assert retrieved_data.count_rows() == 10

print("\nSuccessfully ingested and verified the dataset with expanded nested columns.")
print(f"Retrieved {retrieved_data.count_rows()} rows.")
print("Non-binary columns:")


print(table.to_pandas().head(3))
print(table.schema)

# Clean up
shutil.rmtree(output_dir)

import os
# fix the need for this
os._exit(0)

