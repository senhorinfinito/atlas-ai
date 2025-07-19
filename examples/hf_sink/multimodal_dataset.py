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
from datasets import Dataset, Image, Audio
from PIL import Image as PILImage
import numpy as np
import soundfile as sf

import atlas

# Create a dummy multimodal dataset
test_dir = "hf_multimodal_dataset"
os.makedirs(test_dir, exist_ok=True)
image_path = os.path.join(test_dir, "test.png")
audio_path = os.path.join(test_dir, "test.wav")

# Create a dummy image
PILImage.new('RGB', (10, 10), color = 'red').save(image_path)
# Create a dummy audio file
samplerate = 44100
duration = 1
frequency = 440
t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
amplitude = np.iinfo(np.int16).max * 0.5
data = amplitude * np.sin(2. * np.pi * frequency * t)
sf.write(audio_path, data.astype(np.int16), samplerate)

# Create a Hugging Face dataset
data = {"image": [image_path], "audio": [audio_path], "text": ["This is a test."]}
dataset = Dataset.from_dict(data).cast_column("image", Image()).cast_column("audio", Audio())

# Ingest data
output_dir = "hf_multimodal.lance"
atlas.sink(dataset, output_dir)

# Read the data back
sink = atlas.data_sinks.LanceDataSink(output_dir)
retrieved_data = sink.read()
table = retrieved_data.to_table()

# Verify data
print("Schema:", table.schema)
assert "image" in table.schema.names
assert "audio" in table.schema.names
assert "text" in table.schema.names
assert retrieved_data.count_rows() == 1
assert table.column("image")[0].as_py() is not None
assert table.column("audio")[0].as_py() is not None


# Clean up
shutil.rmtree(test_dir)
shutil.rmtree(output_dir)
