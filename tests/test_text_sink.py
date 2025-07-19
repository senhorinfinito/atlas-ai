import os
import unittest

import lance
import pyarrow as pa

from atlas.data_sinks import sink


class TextSinkTest(unittest.TestCase):
    def setUp(self):
        self.text_path = "test.txt"
        self.lance_path = "test_text.lance"
        with open(self.text_path, "w") as f:
            f.write("line 1\n")
            f.write("line 2\n")
            f.write("line 3\n")

    def tearDown(self):
        os.remove(self.text_path)
        if os.path.exists(self.lance_path):
            import shutil
            shutil.rmtree(self.lance_path)

    def test_sink_text(self):
        sink(self.text_path, self.lance_path)
        dataset = lance.dataset(self.lance_path)
        self.assertEqual(dataset.count_rows(), 3)
        table = dataset.to_table()
        self.assertEqual(table.column_names, ["text"])
        self.assertEqual(table.column("text").to_pylist(), ["line 1", "line 2", "line 3"])


if __name__ == "__main__":
    unittest.main()
