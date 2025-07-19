import os
import unittest
import json

import lance
import pyarrow as pa

from atlas.data_sinks import sink


class PairedTextSinkTest(unittest.TestCase):
    def setUp(self):
        self.paired_text_path = "test_paired_text.jsonl"
        self.lance_path = "test_paired_text.lance"
        with open(self.paired_text_path, "w") as f:
            f.write(json.dumps({"sentence1": "s1.1", "sentence2": "s1.2", "label": 1.0}) + "\n")
            f.write(json.dumps({"sentence1": "s2.1", "sentence2": "s2.2", "label": 0.0}) + "\n")

    def tearDown(self):
        os.remove(self.paired_text_path)
        if os.path.exists(self.lance_path):
            import shutil
            shutil.rmtree(self.lance_path)

    def test_sink_paired_text(self):
        sink(self.paired_text_path, self.lance_path)
        dataset = lance.dataset(self.lance_path)
        self.assertEqual(dataset.count_rows(), 2)
        table = dataset.to_table()
        self.assertEqual(table.column_names, ["sentence1", "sentence2", "label"])
        self.assertEqual(table.column("sentence1").to_pylist(), ["s1.1", "s2.1"])
        self.assertEqual(table.column("sentence2").to_pylist(), ["s1.2", "s2.2"])
        self.assertEqual(table.column("label").to_pylist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
