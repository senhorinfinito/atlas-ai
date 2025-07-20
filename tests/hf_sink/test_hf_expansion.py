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

import unittest
import os
import shutil
import lance
from datasets import Dataset, Features, Value, ClassLabel, Sequence
import pyarrow as pa

from atlas.data_sinks import LanceDataSink
from atlas.tasks.hf.hf import HFDataset


class TestHFExpansion(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_hf_expansion"
        os.makedirs(self.test_dir, exist_ok=True)
        self.data = [
            {"nested": {"a": 1, "b": "one"}},
            {"nested": {"a": 2, "b": "two", "c": True}},
            {"nested": {"a": 3}},
            {"nested": {}},
            {},
            {"nested": {"a": None, "b": "four"}},
        ]
        self.features = Features({
            "nested": {
                "a": Value("int64"),
                "b": Value("string"),
                "c": Value("bool"),
            }
        })
        self.dataset = Dataset.from_list(self.data, features=self.features)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_expansion_with_nulls_robust(self):
        hf_dataset = HFDataset(self.dataset, expand_level=1, handle_nested_nulls=True)
        uri = os.path.join(self.test_dir, "expansion_robust.lance")
        sink = LanceDataSink(path=uri)
        sink.write(hf_dataset)

        retrieved_data = lance.dataset(uri)
        self.assertEqual(retrieved_data.count_rows(), len(self.data))
        
        expected_schema_names = {"nested_a", "nested_b", "nested_c"}
        self.assertEqual(set(retrieved_data.schema.names), expected_schema_names)

        table = retrieved_data.to_table()
        self.assertListEqual(table.column("nested_a").to_pylist(), [1, 2, 3, None, None, None])
        self.assertListEqual(table.column("nested_b").to_pylist(), ["one", "two", None, None, None, "four"])
        self.assertListEqual(table.column("nested_c").to_pylist(), [None, True, None, None, None, None])

    def test_expansion_with_nulls_fast_path_incorrect(self):
        hf_dataset = HFDataset(self.dataset, expand_level=1, handle_nested_nulls=False)
        uri = os.path.join(self.test_dir, "expansion_fast_incorrect.lance")
        sink = LanceDataSink(path=uri)
        sink.write(hf_dataset)

        retrieved_data = lance.dataset(uri)
        table = retrieved_data.to_table()
        
        # The fast path produces incorrect data for missing values.
        self.assertNotEqual(table.column("nested_a").to_pylist(), [1, 2, 3, None, None, None])
        # Specifically, it should produce a 0 for the missing integer value.
        self.assertEqual(table.column("nested_a").to_pylist()[4], 0)


if __name__ == '__main__':
    unittest.main()
