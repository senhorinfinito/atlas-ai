import os
import unittest
import json

import lance
import pyarrow as pa

from atlas.data_sinks import sink


class InstructionSinkTest(unittest.TestCase):
    def setUp(self):
        self.instruction_path = "test.jsonl"
        self.lance_path = "test_instruction.lance"
        with open(self.instruction_path, "w") as f:
            f.write(json.dumps({"instruction": "i1", "input": "i1", "output": "o1"}) + "\n")
            f.write(json.dumps({"instruction": "i2", "input": "i2", "output": "o2"}) + "\n")
            f.write(json.dumps({"instruction": "i3", "input": "i3", "output": "o3"}) + "\n")

    def tearDown(self):
        os.remove(self.instruction_path)
        if os.path.exists(self.lance_path):
            import shutil
            shutil.rmtree(self.lance_path)

    def test_sink_instruction(self):
        sink(self.instruction_path, self.lance_path)
        dataset = lance.dataset(self.lance_path)
        self.assertEqual(dataset.count_rows(), 3)
        table = dataset.to_table()
        self.assertEqual(table.column_names, ["instruction", "input", "output"])
        self.assertEqual(table.column("instruction").to_pylist(), ["i1", "i2", "i3"])
        self.assertEqual(table.column("input").to_pylist(), ["i1", "i2", "i3"])
        self.assertEqual(table.column("output").to_pylist(), ["o1", "o2", "o3"])


if __name__ == "__main__":
    unittest.main()
