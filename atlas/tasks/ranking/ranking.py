# Atlas: A data-centric AI framework
#
# Copyright (c) 2024-present, Atlas Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Generator

import pyarrow as pa

from atlas.tasks.data_model.base import BaseDataset


class RankingDataset(BaseDataset):
    """
    A dataset that reads data from a JSONL file, where each line is a JSON object
    with "query" and "documents" fields.
    """

    def to_batches(self, batch_size: int = 1024, **kwargs) -> Generator[pa.RecordBatch, None, None]:
        """
        Yields batches of the dataset as Arrow RecordBatches.
        """
        queries, documents = [], []

        def process_record(record):
            queries.append(record.get("query", ""))
            # The ms_marco dataset has a 'passages' field, so we'll check for that
            docs = record.get("documents", []) or record.get("passages", {}).get("passage_text", [])
            documents.append(docs)

        if isinstance(self.data, str):
            with open(self.data, "r") as f:
                for line in f:
                    record = json.loads(line)
                    process_record(record)
                    if len(queries) == batch_size:
                        yield pa.RecordBatch.from_arrays(
                            [
                                pa.array(queries, type=pa.string()),
                                pa.array(documents, type=pa.list_(pa.string())),
                            ],
                            schema=self.schema,
                        )
                        queries, documents = [], []
        else:
            for record in self.data:
                process_record(record)
                if len(queries) == batch_size:
                    yield pa.RecordBatch.from_arrays(
                        [
                            pa.array(queries, type=pa.string()),
                            pa.array(documents, type=pa.list_(pa.string())),
                        ],
                        schema=self.schema,
                    )
                    queries, documents = [], []

        if queries:
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(queries, type=pa.string()),
                    pa.array(documents, type=pa.list_(pa.string())),
                ],
                schema=self.schema,
            )

    @property
    def schema(self) -> pa.Schema:
        """
        Returns the schema of the dataset.
        """
        return pa.schema(
            [
                pa.field("query", pa.string()),
                pa.field("documents", pa.list_(pa.string())),
            ]
        )
