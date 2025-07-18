import os
import unittest
import json

import lance
import pyarrow as pa

from atlas.data_sinks import sink


class RankingSinkTest(unittest.TestCase):
    def setUp(self):
        self.ranking_path = "test_ranking.jsonl"
        self.lance_path = "test_ranking.lance"
        with open(self.ranking_path, "w") as f:
            f.write(json.dumps({"query": "q1", "documents": ["d1.1", "d1.2"]}) + "\n")
            f.write(json.dumps({"query": "q2", "documents": ["d2.1", "d2.2"]}) + "\n")

    def tearDown(self):
        os.remove(self.ranking_path)
        if os.path.exists(self.lance_path):
            import shutil
            shutil.rmtree(self.lance_path)

    def test_sink_ranking(self):
        sink(self.ranking_path, self.lance_path)
        dataset = lance.dataset(self.lance_path)
        self.assertEqual(dataset.count_rows(), 2)
        table = dataset.to_table()
        self.assertEqual(table.column_names, ["query", "documents"])
        self.assertEqual(table.column("query").to_pylist(), ["q1", "q2"])
        self.assertEqual(table.column("documents").to_pylist(), [["d1.1", "d1.2"], ["d2.1", "d2.2"]])


if __name__ == "__main__":
    unittest.main()
