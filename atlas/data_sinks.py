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

import os
import json
from typing import Any, Dict, Optional, Union

import lance
import pyarrow as pa

from atlas.tasks.data_model.base import BaseDataset
from atlas.tasks.data_model.factory import create_dataset
from datasets import Dataset


class LanceDataSink:
    def __init__(self, path: str, mode: str = "overwrite", **kwargs):
        self.path = path
        self.mode = mode
        self.kwargs = kwargs
        self._metadata = None

    def write(self, data: Union[str, BaseDataset, Dataset], task: Optional[str] = None, format: Optional[str] = None, **kwargs):
        if isinstance(data, Dataset) and task is None:
            task = "hf"

        if isinstance(data, str):
            dataset = create_dataset(data, task=task, format=format, **kwargs)
        elif isinstance(data, Dataset) or (
            hasattr(data, "__iter__") and hasattr(data, "__next__")
        ):
            dataset = create_dataset(data, task=task, format=format, **kwargs)
        else:
            dataset = data
        
        self._metadata = dataset.metadata
        dataset.to_lance(self.path, mode=self.mode, **self.kwargs)

    def read(self):
        return lance.dataset(self.path)

    @property
    def metadata(self):
        if self._metadata:
            return self._metadata
        if os.path.exists(self.path):
            dataset = lance.dataset(self.path)
            schema_metadata = dataset.schema.metadata
            if schema_metadata:
                self._metadata = {k.decode(): v.decode() for k, v in schema_metadata.items()}
                if 'decode_meta' in self._metadata:
                    self._metadata['decode_meta'] = json.loads(self._metadata['decode_meta'])
        return self._metadata


def sink(
    data: Union[str, BaseDataset, Dataset],
    uri: Optional[str] = None,
    task: Optional[str] = None,
    format: Optional[str] = None,
    mode: str = "overwrite",
    **kwargs,
):
    if not uri:
        raise ValueError("URI must be specified for the sink operation.")
    sink = LanceDataSink(path=uri, mode=mode, **kwargs)
    sink.write(data, task=task, format=format, **kwargs)
