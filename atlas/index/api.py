from typing import Any, Dict, List, Optional
import json
import io

import lance
import lancedb
import pyarrow as pa
from rich.table import Table
from rich.console import Console
from PIL import Image

from .vectorizer.vectorizer import Vectorizer


class Indexer:
    """
    A class to manage the indexing of data in a LanceDB table.
    """

    def __init__(self, uri: str):
        """
        Initializes the Indexer.

        Args:
            uri (str): The URI of the Lance dataset.
        """
        self.uri = uri
        db_path = uri.rsplit("/", 1)[0]
        table_name = uri.rsplit("/", 1)[-1].replace(".lance", "")
        
        self.db = lancedb.connect(db_path)
        
        if table_name in self.db.table_names():
            self.table = self.db.open_table(table_name)
        else:
            self.table = self.db.create_table(table_name, data=self.uri)


    def _get_modality(self, column: str) -> str:
        """
        Determines the modality of a column based on the table's schema metadata.
        """
        # LanceDB now stores metadata in a different way, let's access the lance_dataset schema
        lance_dataset = lance.dataset(self.uri)
        if lance_dataset.schema.metadata and b"decode_meta" in lance_dataset.schema.metadata:
            decode_meta = json.loads(lance_dataset.schema.metadata[b"decode_meta"])
            if column in decode_meta:
                if "Image" in decode_meta[column]:
                    return "image"
                if "Audio" in decode_meta[column]:
                    return "audio"
        # Default to text if no specific modality is found
        return "text"

    def create_index(
        self,
        column: str,
        index_type: str,
        model: Optional[Any] = None,
        vector_column_name: str = "vector",
        **kwargs,
    ):
        """
        Creates an index on a specified column.

        Args:
            column (str): The name of the column to index.
            index_type (str): The type of index to create ('vector' or 'fts').
            model (Optional[Any]): The embedding model to use for vector indexing.
                                   If not provided, a default model will be used
                                   based on the column's data type.
            vector_column_name (str): The name of the column to store the vectors in.
            **kwargs: Additional keyword arguments for index creation.
        """
        if index_type == "vector":
            modality = self._get_modality(column)
            if model is None:
                vectorizer = Vectorizer(modality=modality)
            else:
                vectorizer = Vectorizer(model_name=model, modality=modality)

            # Read the data from the column
            data = self.table.to_pandas()[column].tolist()

            if modality == "image":
                # Decode binary data to images
                data = [Image.open(io.BytesIO(d)) for d in data]

            embeddings = vectorizer.vectorize(data)

            # Convert table to pandas, add the new column, and overwrite the table
            df = self.table.to_pandas()
            df[vector_column_name] = embeddings
            
            self.db.drop_table(self.table.name)
            self.table = self.db.create_table(self.table.name, data=df)

            print(f"Creating vector index on column '{vector_column_name}'...")
            self.table.create_index(
                vector_column_name=vector_column_name,
            )
        elif index_type == "fts":
            print(f"Creating FTS index on column '{column}'...")
            self.table.create_fts_index(column, **kwargs)
        else:
            raise ValueError("index_type must be either 'vector' or 'fts'")

    def list_indexes(self, column: Optional[str] = None):
        """
        Displays the table schema with existing index types for each column.

        Args:
            column (Optional[str]): If provided, only show information for this column.
        """
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Column Name", style="dim")
        table.add_column("Data Type")
        table.add_column("Index Type")

        schema = self.table.schema
        
        print("--- DEBUG: Raw indices ---")
        indices = self.table.list_indices()
        print(indices)
        print("--- END DEBUG ---")

        try:
            indexed_columns = {
                " ".join(idx['columns']): idx.name for idx in indices
            }
        except (KeyError, IndexError):
            print("couldn't read indeces")
            indexed_columns = {}

        for field in schema:
            if column and field.name != column:
                continue

            index_type = indexed_columns.get(field.name, "None")
            table.add_row(field.name, str(field.type), index_type)

        console.print(table)
