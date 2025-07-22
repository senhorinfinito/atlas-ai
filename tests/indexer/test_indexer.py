import os
import shutil
import uuid

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from lancedb.pydantic import LanceModel, vector

from atlas.index import api as indexer_api

# A temporary directory to store test artifacts
TEST_DIR = "/tmp/atlas_indexer_test"


class MySchema(LanceModel):
    vector: vector(128)
    id: int
    text: str


@pytest.fixture(scope="module")
def lance_dataset():
    """Create a dummy lance dataset for testing."""
    os.makedirs(TEST_DIR, exist_ok=True)
    dataset_path = os.path.join(TEST_DIR, f"{uuid.uuid4()}.lance")
    db_path = os.path.join(TEST_DIR, "lancedb")
    os.makedirs(db_path, exist_ok=True)

    # Create some dummy data with vectors of the correct type
    vectors = [np.random.rand(128).astype("float32") for _ in range(256)]
    ids = list(range(256))
    texts = [f"this is text {i}" for i in range(256)]

    # Use pyarrow to create a table with the correct schema
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 128)),
            pa.field("id", pa.int64()),
            pa.field("text", pa.string()),
        ]
    )
    tbl = pa.Table.from_pydict(
        {"vector": vectors, "id": ids, "text": texts}, schema=schema
    )

    # Save it as a lance dataset
    lance.write_dataset(tbl, dataset_path)
    
    db = indexer_api.lancedb.connect(db_path)
    db.create_table(dataset_path.rsplit("/", 1)[-1].replace(".lance", ""), data=tbl)


    yield dataset_path

    # Cleanup
    shutil.rmtree(TEST_DIR, ignore_errors=True)


def test_indexer_workflow(lance_dataset):
    """
    Tests the full workflow:
    1. Open a lance dataset.
    2. Create a vector index on it.
    3. Create an FTS index on it.
    4. List indexes.
    """
    # 1. Open the dataset and wrap it with the Indexer
    idx = indexer_api.Indexer(lance_dataset)

    # 2. Create the vector index on the 'vector' column
    idx.create_index("vector", "vector", num_partitions=4, num_sub_vectors=64)

    # 3. Create the FTS index on the 'text' column
    idx.create_index("text", "fts")

    # 4. List indexes and verify
    indices = idx.table.list_indices()
    assert len(indices) == 2
    index_names = {idx['name'] for idx in indices}
    assert 'vector_idx' in index_names
    assert 'text_idx' in index_names


def test_list_indexes(capsys, lance_dataset):
    """Tests the list_indexes method."""
    idx = indexer_api.Indexer(lance_dataset)
    idx.create_index("vector", "vector")

    idx.list_indexes()
    captured = capsys.readouterr()
    assert "vector" in captured.out
    assert "vector_idx" in captured.out

    idx.list_indexes(column="vector")
    captured = capsys.readouterr()
    assert "vector" in captured.out
    assert "vector_idx" in captured.out
    assert "id" not in captured.out or "id_idx" not in captured.out
