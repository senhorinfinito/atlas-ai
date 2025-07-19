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
from datasets import Dataset, Audio
import numpy as np
import soundfile as sf

import atlas

# Create a dummy audio dataset
test_dir = "dummy_audio_dataset"
os.makedirs(test_dir, exist_ok=True)
audio_path = os.path.join(test_dir, "test.wav")

# Create a dummy audio file
samplerate = 44100
duration = 1
frequency = 440
t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
amplitude = np.iinfo(np.int16).max * 0.5
data = amplitude * np.sin(2. * np.pi * frequency * t)
sf.write(audio_path, data.astype(np.int16), samplerate)

# Create a Hugging Face dataset
data = {"audio": [audio_path]}
dataset = Dataset.from_dict(data).cast_column("audio", Audio())

# Ingest data
output_dir = "dummy_audio.lance"
atlas.sink(dataset, output_dir)

# Read the data back
sink = atlas.data_sinks.LanceDataSink(output_dir)
retrieved_data = sink.read()
table = retrieved_data.to_table()

# Verify data
print("Schema:", table.schema)
print("Data:", table)

# Clean up
shutil.rmtree(test_dir)
shutil.rmtree(output_dir)