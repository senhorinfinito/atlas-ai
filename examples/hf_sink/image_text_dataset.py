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
from datasets import load_dataset

import atlas

# Load a small subset of an image-text dataset
dataset = load_dataset("jxie/flickr8k", split="train[:10]")

# Ingest data
output_dir = "image_text.lance"
atlas.sink(dataset, output_dir)

# Read the data back
sink = atlas.data_sinks.LanceDataSink(output_dir)
retrieved_data = sink.read()
table = retrieved_data.to_table()

# Verify data
print("Schema:", table.schema)
assert "image" in table.schema.names
assert "caption_0" in table.schema.names
assert retrieved_data.count_rows() == 10
assert table.column("image")[0].as_py() is not None

# Clean up
shutil.rmtree(output_dir)
