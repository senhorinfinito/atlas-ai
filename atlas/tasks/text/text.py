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

from typing import Generator

import pyarrow as pa

from atlas.tasks.data_model.base import BaseDataset


class TextDataset(BaseDataset):
    """
    A dataset that reads data from a text file, where each line is a record.
    """

    def to_batches(self, batch_size: int = 1024, **kwargs) -> Generator[pa.RecordBatch, None, None]:
        """
        Yields batches of the dataset as Arrow RecordBatches.
        """
        with open(self.data, "r") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
                if len(lines) == batch_size:
                    yield pa.RecordBatch.from_arrays(
                        [pa.array(lines, type=pa.string())],
                        names=["text"],
                    )
                    lines = []
            if lines:
                yield pa.RecordBatch.from_arrays(
                    [pa.array(lines, type=pa.string())],
                    names=["text"],
                )

    @property
    def schema(self) -> pa.Schema:
        """
        Returns the schema of the dataset.
        """
        return pa.schema([pa.field("text", pa.string())])
