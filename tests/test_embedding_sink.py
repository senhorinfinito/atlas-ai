import os
import unittest
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import lance

from atlas.data_sinks import sink


class EmbeddingSinkTest(unittest.TestCase):
    def setUp(self):
        self.embedding_path = "test.parquet"
        self.lance_path = "test_embedding.lance"
        dummy_embedding_content = {
            "text": ["text1", "text2", "text3"],
            "embedding": [
                np.random.rand(5).astype(np.float32),
                np.random.rand(5).astype(np.float32),
                np.random.rand(5).astype(np.float32),
            ],
        }
        df = pd.DataFrame(dummy_embedding_content)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.embedding_path)

    def tearDown(self):
        os.remove(self.embedding_path)
        if os.path.exists(self.lance_path):
            import shutil
            shutil.rmtree(self.lance_path)

    def test_sink_embedding(self):
        sink(self.embedding_path, self.lance_path)
        dataset = lance.dataset(self.lance_path)
        self.assertEqual(dataset.count_rows(), 3)
        table = dataset.to_table()
        self.assertEqual(table.column_names, ["text", "embedding"])
        self.assertEqual(table.column("text").to_pylist(), ["text1", "text2", "text3"])


if __name__ == "__main__":
    unittest.main()
