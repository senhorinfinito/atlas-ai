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


class CoTDataset(BaseDataset):
    """
    A dataset that reads data from a JSONL file, where each line is a JSON object
    with "question", "thought", and "answer" fields.
    """

    def to_batches(self, batch_size: int = 1024, **kwargs) -> Generator[pa.RecordBatch, None, None]:
        """
        Yields batches of the dataset as Arrow RecordBatches.
        """
        questions, thoughts, answers = [], [], []

        def process_record(record):
            questions.append(record.get("question", ""))
            thoughts.append(record.get("thought", ""))
            answers.append(record.get("answer", ""))

        if isinstance(self.data, str):
            with open(self.data, "r") as f:
                for line in f:
                    record = json.loads(line)
                    process_record(record)
                    if len(questions) == batch_size:
                        yield pa.RecordBatch.from_arrays(
                            [
                                pa.array(questions, type=pa.string()),
                                pa.array(thoughts, type=pa.string()),
                                pa.array(answers, type=pa.string()),
                            ],
                            schema=self.schema,
                        )
                        questions, thoughts, answers = [], [], []
        else:
            for record in self.data:
                process_record(record)
                if len(questions) == batch_size:
                    yield pa.RecordBatch.from_arrays(
                        [
                            pa.array(questions, type=pa.string()),
                            pa.array(thoughts, type=pa.string()),
                            pa.array(answers, type=pa.string()),
                        ],
                        schema=self.schema,
                    )
                    questions, thoughts, answers = [], [], []

        if questions:
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(questions, type=pa.string()),
                    pa.array(thoughts, type=pa.string()),
                    pa.array(answers, type=pa.string()),
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
                pa.field("question", pa.string()),
                pa.field("thought", pa.string()),
                pa.field("answer", pa.string()),
            ]
        )
