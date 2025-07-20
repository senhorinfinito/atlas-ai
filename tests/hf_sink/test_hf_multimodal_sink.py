# Atlas: A data-centric AI framework
#
# Copyright (c) 2024-present, Atlas Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os
import shutil
import json
import lance
from datasets import Dataset, Image, Audio
from PIL import Image as PILImage
import numpy as np
import pyarrow as pa

from atlas.data_sinks import sink


class HFMultimodalDataSinkTest(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_hf_multimodal_sink"
        os.makedirs(self.test_dir, exist_ok=True)
        self.image_path = os.path.join(self.test_dir, "test.png")
        self.audio_path = os.path.join(self.test_dir, "test.wav")
        # Create a dummy image
        PILImage.new('RGB', (10, 10), color = 'red').save(self.image_path)
        # Create a dummy audio file
        import soundfile as sf
        samplerate = 44100
        duration = 1
        frequency = 440
        t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        sf.write(self.audio_path, data.astype(np.int16), samplerate)


    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_image_ingestion_retrieval(self):
        data = {"image": [self.image_path]}
        dataset = Dataset.from_dict(data).cast_column("image", Image())
        uri = os.path.join(self.test_dir, "image.lance")
        
        # Ingest data
        sink(dataset, uri, task="hf")

        # Retrieve data
        retrieved_data = lance.dataset(uri)
        self.assertIn("image", retrieved_data.schema.names)
        
        # Verify decode_meta
        schema_metadata = retrieved_data.schema.metadata
        self.assertIn(b'decode_meta', schema_metadata)
        decode_meta = json.loads(schema_metadata[b'decode_meta'])
        self.assertIn("image", decode_meta)
        self.assertEqual(decode_meta["image"], str(Image()))

        # Verify image data
        table = retrieved_data.to_table()
        retrieved_image_bytes = table.column("image")[0].as_py()
        with open(self.image_path, "rb") as f:
            original_image_bytes = f.read()
        self.assertEqual(retrieved_image_bytes, original_image_bytes)

    def test_audio_ingestion_retrieval(self):
        data = {"audio": [self.audio_path]}
        dataset = Dataset.from_dict(data).cast_column("audio", Audio())
        uri = os.path.join(self.test_dir, "audio.lance")

        # Ingest data
        sink(dataset, uri, task="hf")

        # Retrieve data
        retrieved_data = lance.dataset(uri)
        self.assertIn("audio", retrieved_data.schema.names)

        # Verify decode_meta
        schema_metadata = retrieved_data.schema.metadata
        self.assertIn(b'decode_meta', schema_metadata)
        decode_meta = json.loads(schema_metadata[b'decode_meta'])
        self.assertIn("audio", decode_meta)
        self.assertEqual(decode_meta["audio"], str(Audio()))

        # Verify audio data
        table = retrieved_data.to_table()
        retrieved_audio_bytes = table.column("audio")[0].as_py()
        with open(self.audio_path, "rb") as f:
            original_audio_bytes = f.read()
        self.assertEqual(retrieved_audio_bytes, original_audio_bytes)

if __name__ == '__main__':
    unittest.main()