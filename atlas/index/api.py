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
            data = lance.dataset(self.uri)
            self.table = self.db.create_table(table_name, data=data)


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
        batch_size: int = 32,
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
            batch_size (int): The batch size for vectorization.
            **kwargs: Additional keyword arguments for index creation.
        """
        if index_type == "vector":
            # Check if the column is a pre-computed vector
            if column in self.table.schema.names:
                field = self.table.schema.field(column)
                if pa.types.is_fixed_size_list(field.type) and pa.types.is_floating(
                    field.type.value_type
                ):
                    print(
                        f"Creating vector index on pre-computed vectors in column '{column}'..."
                    )
                    self.table.create_index(vector_column_name=column, **kwargs)
                    return

            # If not a pre-computed vector, vectorize the source column
            modality = self._get_modality(column)
            vectorizer = Vectorizer(model_name=model, modality=modality)

            # TODO: Implement an auto-batcher to dynamically determine the batch size
            temp_table_name = f"{self.table.name}_temp_embeddings"
            if temp_table_name in self.db.table_names():
                self.db.drop_table(temp_table_name)

            scanner = self.table.to_lance().scanner(
                with_row_id=True, batch_size=batch_size
            )

            first_batch = True
            for batch in scanner.to_batches():
                column_data = batch.column(column).to_pylist()
                embeddings = vectorizer.vectorize(column_data, batch_size=batch_size)

                embedding_table = pa.Table.from_pydict(
                    {
                        "_rowid": batch.column("_rowid"),
                        vector_column_name: embeddings,
                    }
                )

                if first_batch:
                    self.db.create_table(temp_table_name, embedding_table)
                    first_batch = False
                else:
                    temp_table = self.db.open_table(temp_table_name)
                    temp_table.add(embedding_table)

            if first_batch:
                print("No data to index.")
                return

            temp_table = self.db.open_table(temp_table_name)
            self.table.merge(temp_table, left_on="_rowid", right_on="_rowid")

            self.db.drop_table(temp_table_name)

            print(f"Creating vector index on column '{vector_column_name}'...")
            self.table.create_index(
                vector_column_name=vector_column_name,
                **kwargs,
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
        
        indices = self.table.list_indices()

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
