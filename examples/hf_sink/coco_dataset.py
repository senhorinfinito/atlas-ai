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
from datasets import Dataset, Image, Features, Value, Sequence
from PIL import Image as PILImage

import atlas

# Create a dummy COCO-style dataset
test_dir = "dummy_coco_dataset"
os.makedirs(test_dir, exist_ok=True)
image_path = os.path.join(test_dir, "test.png")
PILImage.new('RGB', (10, 10), color = 'red').save(image_path)

features = Features({
    'image': Image(),
    'annotations': Sequence({
        'bbox': Sequence(Value(dtype='float64')),
        'label': Value(dtype='string'),
    })
})

data = {
    "image": [image_path] * 2,
    "annotations": [
        [{"bbox": [10, 20, 30, 40], "label": "cat"}],
        [{"bbox": [50, 60, 70, 80], "label": "dog"}, {"bbox": [90, 100, 110, 120], "label": "person"}],
    ]
}
dataset = Dataset.from_dict(data, features=features)

# Ingest data
output_dir = "coco_style.lance"
atlas.sink(dataset, output_dir)

# Read the data back
sink = atlas.data_sinks.LanceDataSink(output_dir)
retrieved_data = sink.read()
table = retrieved_data.to_table()

# Verify data
print("Schema:", table.schema)
assert "annotations" in table.schema.names
assert "bbox" in table.schema.field("annotations").type.value_type.names
assert "label" in table.schema.field("annotations").type.value_type.names
assert retrieved_data.count_rows() == 2
assert len(table.column("annotations")[1].as_py()) == 2

# Clean up
shutil.rmtree(test_dir)
shutil.rmtree(output_dir)
