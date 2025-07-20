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

import os
import shutil
from datasets import load_dataset, Dataset, Features, Sequence, Value, Image
import pyarrow as pa
from PIL import Image as PILImage

import atlas
from atlas.tasks.hf.hf import HFDataset

# 1. Load a real, public dataset with images
base_dataset = load_dataset("jxie/flickr8k", split="train[:10]")

# 2. Create a new dataset with a COCO-style nested column
images = []
bboxes = []
categories = []
for i, example in enumerate(base_dataset):
    images.append(example['image'])
    if i % 2 == 0:
        bboxes.append([[10.0, 20.0, 30.0, 40.0]])
        categories.append(['cat'])
    else:
        bboxes.append([[50.0, 60.0, 70.0, 80.0], [90.0, 100.0, 110.0, 120.0]])
        categories.append(['dog', 'person'])

new_features = Features({
    'image': Image(),
    'bbox': Sequence(Sequence(Value(dtype='float64'))),
    'category': Sequence(Value(dtype='string')),
})

dataset = Dataset.from_dict({"image": images, "bbox": bboxes, "category": categories}, features=new_features)


# 3. Instantiate HFDataset with expansion enabled (but it should not expand sequences)
hf_dataset = HFDataset(dataset, expand_level=1)

# 4. Ingest data
output_dir = "coco_not_expanded.lance"
atlas.sink(hf_dataset, output_dir)

# 5. Read the data back
sink = atlas.data_sinks.LanceDataSink(output_dir)
retrieved_data = sink.read()
table = retrieved_data.to_table()

# 6. Verify the schema and data
print("Schema:", table.schema)

# Verify that the 'image' column was ingested as binary
assert pa.types.is_large_binary(table.schema.field("image").type)

# Verify that the nested 'bbox' and 'category' Sequences were NOT expanded
assert "bbox" in table.schema.names
assert pa.types.is_list(table.schema.field("bbox").type)
assert "category" in table.schema.names
assert pa.types.is_list(table.schema.field("category").type)


# Verify row count
assert retrieved_data.count_rows() == 10

print("\nSuccessfully ingested and verified the dataset with non-expanded nested columns.")
print(f"Retrieved {retrieved_data.count_rows()} rows.")
print("Non-binary columns:")
for col_name in [name for name in table.schema.names if "image" not in name]:
    print(f"- {col_name}: {table[col_name].slice(0, 5)}")


# Clean up
shutil.rmtree(output_dir)