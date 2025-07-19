import os
import unittest
import json

import lance
import pyarrow as pa

from atlas.data_sinks import sink


class CoTSinkTest(unittest.TestCase):
    def setUp(self):
        self.cot_path = "test_cot.jsonl"
        self.lance_path = "test_cot.lance"
        with open(self.cot_path, "w") as f:
            f.write(json.dumps({"question": "q1", "thought": "t1", "answer": "a1"}) + "\n")
            f.write(json.dumps({"question": "q2", "thought": "t2", "answer": "a2"}) + "\n")

    def tearDown(self):
        os.remove(self.cot_path)
        if os.path.exists(self.lance_path):
            import shutil
            shutil.rmtree(self.lance_path)

    def test_sink_cot(self):
        sink(self.cot_path, self.lance_path)
        dataset = lance.dataset(self.lance_path)
        self.assertEqual(dataset.count_rows(), 2)
        table = dataset.to_table()
        self.assertEqual(table.column_names, ["question", "thought", "answer"])
        self.assertEqual(table.column("question").to_pylist(), ["q1", "q2"])
        self.assertEqual(table.column("thought").to_pylist(), ["t1", "t2"])
        self.assertEqual(table.column("answer").to_pylist(), ["a1", "a2"])


if __name__ == "__main__":
    unittest.main()
