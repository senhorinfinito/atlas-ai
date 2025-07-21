import os
import lance
import lancedb
import pandas as pd
from lancedb.pydantic import LanceModel, vector
from atlas.index.api import Indexer


class MySchema(LanceModel):
    vector: vector(128)
    text: str
    id: int


def main():
    # Create a dummy Lance dataset
    if os.path.exists("my_table.lance"):
        os.system("rm -rf my_table.lance")

    # Create an Indexer
    indexer = Indexer(
        uri="my_table.lance",
        schema=MySchema,
        index_type="vector",
    )

    # Add some data
    data = [
        {"vector": [0.1] * 128, "text": "hello world", "id": 1},
        {"vector": [0.2] * 128, "text": "hello mars", "id": 2},
    ]
    indexer.add(data)

    # Create the index
    indexer.create_index()

    # Perform a search
    query_vector = [0.15] * 128
    results = indexer.search(query_vector, limit=1)
    print("Vector search results:")
    print(results)

    # Create an FTS Indexer
    indexer_fts = Indexer(
        uri="my_table.lance",
        schema=MySchema,
        index_type="fts",
        text_column_name="text",
    )

    # Create the FTS index
    indexer_fts.create_index()

    # Perform an FTS search
    results_fts = indexer_fts.search("hello", limit=2)
    print("\nFTS search results:")
    print(results_fts)


if __name__ == "__main__":
    main()