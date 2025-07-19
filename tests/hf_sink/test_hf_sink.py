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
import unittest
import lance
from datasets import load_dataset

from atlas.data_sinks import sink


class TestHFSink(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset("glue", "mrpc", split="train")
        self.uri = "mrpc.lance"

    def tearDown(self):
        if os.path.exists(self.uri):
            import shutil
            shutil.rmtree(self.uri)

    def test_hf_sink(self):
        sink(self.dataset, self.uri)
        self.assertTrue(os.path.exists(self.uri))
        lance_dataset = lance.dataset(self.uri)
        self.assertEqual(lance_dataset.count_rows(), len(self.dataset))
        self.assertEqual(lance_dataset.schema, self.dataset.features.arrow_schema)


if __name__ == "__main__":
    unittest.main()
