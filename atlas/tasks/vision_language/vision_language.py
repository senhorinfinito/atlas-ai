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
import os
from typing import Generator

import pyarrow as pa

from atlas.tasks.data_model.base import BaseDataset


class VisionLanguageDataset(BaseDataset):
    """
    A dataset that reads data from a JSONL file, where each line is a JSON object
    with "image" (path) and "text" fields.
    """

    def to_batches(self, batch_size: int = 1024, **kwargs) -> Generator[pa.RecordBatch, None, None]:
        """
        Yields batches of the dataset as Arrow RecordBatches.
        """
        with open(self.data, "r") as f:
            images, texts = [], []
            for line in f:
                record = json.loads(line)
                image_path = record.get("image")
                if image_path and os.path.exists(image_path):
                    with open(image_path, "rb") as img_f:
                        images.append(img_f.read())
                else:
                    images.append(None)
                texts.append(record.get("text", ""))

                if len(images) == batch_size:
                    yield pa.RecordBatch.from_arrays(
                        [
                            pa.array(images, type=pa.binary()),
                            pa.array(texts, type=pa.string()),
                        ],
                        names=["image", "text"],
                    )
                    images, texts = [], []
            if images:
                yield pa.RecordBatch.from_arrays(
                    [
                        pa.array(images, type=pa.binary()),
                        pa.array(texts, type=pa.string()),
                    ],
                    names=["image", "text"],
                )

    @property
    def schema(self) -> pa.Schema:
        """
        Returns the schema of the dataset.
        """
        return pa.schema(
            [
                pa.field("image", pa.binary()),
                pa.field("text", pa.string()),
            ]
        )
