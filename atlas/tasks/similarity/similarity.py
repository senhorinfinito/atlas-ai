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

from typing import Generator, Dict, Any

import pyarrow as pa

from atlas.tasks.data_model.base import BaseDataset


import json

class SimilarityDataset(BaseDataset):
    """
    A dataset for sentence similarity tasks.
    """

    def to_batches(self, batch_size: int = 1024, **kwargs) -> Generator[pa.RecordBatch, None, None]:
        """
        Yields batches of the dataset as Arrow RecordBatches.
        """
        sentence1_list, sentence2_list, similarity_score_list = [], [], []
        
        # If self.data is a path to a file, open it and read line by line
        if isinstance(self.data, str):
            with open(self.data, "r") as f:
                for line in f:
                    record = json.loads(line)
                    sentence1_list.append(record["sentence1"])
                    sentence2_list.append(record["sentence2"])
                    similarity_score_list.append(record["similarity_score"])
                    if len(sentence1_list) == batch_size:
                        yield pa.RecordBatch.from_arrays(
                            [
                                pa.array(sentence1_list, type=pa.string()),
                                pa.array(sentence2_list, type=pa.string()),
                                pa.array(similarity_score_list, type=pa.float32()),
                            ],
                            schema=self.schema,
                        )
                        sentence1_list, sentence2_list, similarity_score_list = [], [], []
        else: # self.data is a generator
            for record in self.data:
                sentence1_list.append(record["sentence1"])
                sentence2_list.append(record["sentence2"])
                similarity_score_list.append(record["similarity_score"])
                if len(sentence1_list) == batch_size:
                    yield pa.RecordBatch.from_arrays(
                        [
                            pa.array(sentence1_list, type=pa.string()),
                            pa.array(sentence2_list, type=pa.string()),
                            pa.array(similarity_score_list, type=pa.float32()),
                        ],
                        schema=self.schema,
                    )
                    sentence1_list, sentence2_list, similarity_score_list = [], [], []

        if sentence1_list:
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(sentence1_list, type=pa.string()),
                    pa.array(sentence2_list, type=pa.string()),
                    pa.array(similarity_score_list, type=pa.float32()),
                ],
                schema=self.schema,
            )

    @property
    def schema(self) -> pa.Schema:
        """
        Returns the schema of the dataset.
        """
        return pa.schema([
            pa.field("sentence1", pa.string()),
            pa.field("sentence2", pa.string()),
            pa.field("similarity_score", pa.float32()),
        ])
