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


@pytest.fixture(scope="function")
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
    5. Verify indexes with searches.
    """
    # 1. Open the dataset and wrap it with the Indexer
    idx = indexer_api.Indexer(lance_dataset)
    original_schema = idx.table.schema
    original_count = idx.table.to_pandas().shape[0]

    # 2. Create the vector index on the 'text' column, storing vectors in 'text_embeddings'
    vector_column_name = "text_embeddings"
    idx.create_index(
        "text",
        "vector",
        vector_column_name=vector_column_name,
        num_partitions=4,
        num_sub_vectors=64,
    )

    # Verify the new column exists and other columns are preserved
    assert vector_column_name in idx.table.schema.names
    assert all(field.name in idx.table.schema.names for field in original_schema)
    assert idx.table.to_pandas().shape[0] == original_count

    # Verify the new column contains valid vectors
    table_head = idx.table.search(vector_column_name=vector_column_name).limit(5).to_pandas()
    assert not table_head[vector_column_name].isnull().any()
    assert isinstance(table_head[vector_column_name][0], np.ndarray)

    # 3. Create the FTS index on the 'text' column
    idx.create_index("text", "fts")

    # 4. List indexes and verify
    indices = idx.table.list_indices()
    assert len(indices) == 2
    index_names = {idx.name for idx in indices}
    assert "text_embeddings_idx" in index_names
    assert "text_idx" in index_names

    # 5. Verify indexes with searches
    # Vector search
    query_vector = table_head[vector_column_name][0]
    search_results = idx.table.search(query_vector, vector_column_name=vector_column_name).limit(1).to_pandas()
    assert search_results.shape[0] > 0
    assert search_results['text'][0] == table_head['text'][0]

    # FTS search
    fts_results = idx.table.search("text", query_type="fts").limit(5).to_pandas()
    assert fts_results.shape[0] > 0


def test_indexer_precomputed_vector_workflow(lance_dataset):
    """
    Tests the workflow for creating an index on a pre-computed vector column.
    """
    # 1. Open the dataset and wrap it with the Indexer
    idx = indexer_api.Indexer(lance_dataset)
    original_schema = idx.table.schema
    original_count = idx.table.to_pandas().shape[0]

    # 2. Create the vector index on the pre-computed 'vector' column
    idx.create_index("vector", "vector")

    # Verify the schema and count are unchanged
    assert idx.table.schema == original_schema
    assert idx.table.to_pandas().shape[0] == original_count

    # 3. List indexes and verify
    indices = idx.table.list_indices()
    assert len(indices) == 1
    assert indices[0].name == "vector_idx"

    # 4. Verify the index with a search
    table_head = idx.table.search().limit(1).to_pandas()
    query_vector = table_head["vector"][0]
    search_results = idx.table.search(query_vector, vector_column_name="vector").limit(1).to_pandas()
    assert search_results.shape[0] > 0
    assert search_results['id'][0] == table_head['id'][0]


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