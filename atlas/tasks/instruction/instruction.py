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


class InstructionDataset(BaseDataset):
    """
    A dataset that reads data from a JSONL file, where each line is a JSON object
    with "instruction", "input", and "output" fields.
    """

    def to_batches(self, batch_size: int = 1024, **kwargs) -> Generator[pa.RecordBatch, None, None]:
        """
        Yields batches of the dataset as Arrow RecordBatches.
        """
        instructions, inputs, outputs, responses = [], [], [], []

        def process_record(record):
            instructions.append(record.get("instruction", ""))
            inputs.append(record.get("input", "") or record.get("context", ""))
            outputs.append(record.get("output", ""))
            responses.append(record.get("response", ""))


        if isinstance(self.data, str):
            with open(self.data, "r") as f:
                for line in f:
                    record = json.loads(line)
                    process_record(record)
                    if len(instructions) == batch_size:
                        yield pa.RecordBatch.from_arrays(
                            [
                                pa.array(instructions, type=pa.string()),
                                pa.array(inputs, type=pa.string()),
                                pa.array(outputs, type=pa.string()),
                                pa.array(responses, type=pa.string()),
                            ],
                            schema=self.schema,
                        )
                        instructions, inputs, outputs, responses = [], [], [], []
        else:
            for record in self.data:
                process_record(record)
                if len(instructions) == batch_size:
                    yield pa.RecordBatch.from_arrays(
                        [
                            pa.array(instructions, type=pa.string()),
                            pa.array(inputs, type=pa.string()),
                            pa.array(outputs, type=pa.string()),
                            pa.array(responses, type=pa.string()),
                        ],
                        schema=self.schema,
                    )
                    instructions, inputs, outputs, responses = [], [], [], []

        if instructions:
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(instructions, type=pa.string()),
                    pa.array(inputs, type=pa.string()),
                    pa.array(outputs, type=pa.string()),
                    pa.array(responses, type=pa.string()),
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
                pa.field("instruction", pa.string()),
                pa.field("input", pa.string()),
                pa.field("output", pa.string()),
                pa.field("response", pa.string()),
            ]
        )
