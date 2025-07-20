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
    """
    Ingests data from a source and sinks it into a Lance dataset.

    This is the main entry point for data ingestion in Atlas. It automatically
    infers the data type and format, and then uses the appropriate data loader
    to write the data to a Lance dataset.

    Args:
        data: The data to sink. Can be a path to a file or directory, or a
            Hugging Face Dataset object.
        uri (str): The URI of the Lance dataset to create.
        task (str, optional): The task type of the data (e.g., "object_detection").
            If not provided, Atlas will try to infer it.
        format (str, optional): The format of the data (e.g., "coco").
            If not provided, Atlas will try to infer it.
        mode (str, optional): The write mode for the Lance dataset.
            Defaults to "overwrite".
        **kwargs: Additional options passed to the underlying data loader.

    Keyword Args:
        expand_level (int): For Hugging Face datasets with nested schemas,
            this specifies the level of nesting to expand into separate columns.
            Defaults to 0 (no expansion).
        handle_nested_nulls (bool): For Hugging Face datasets, this flag
            determines how to handle missing or null values in nested fields
            during expansion.
            - False (Default): Prioritizes speed. Missing fields are filled
              with default values for their data type (e.g., 0 for integers,
              "" for strings). This is fast but can lead to inaccurate
              downstream analysis if the distinction between a real default
              value and a missing value is important.
            - True: Prioritizes data correctness. Missing fields are
              represented as true `None` (null) values. This is slightly
              slower but ensures that missing data is not misrepresented,
              leading to more accurate analysis.
    """
    if not uri:
        raise ValueError("URI must be specified for the sink operation.")
        
    # Separate kwargs for dataset creation from Lance write kwargs
    dataset_kwargs = {
        "expand_level": kwargs.pop("expand_level", 0),
        "handle_nested_nulls": kwargs.pop("handle_nested_nulls", False),
    }
    # Pass the remaining kwargs to the LanceDataSink
    sink = LanceDataSink(path=uri, mode=mode, **kwargs)
    sink.write(data, task=task, format=format, **dataset_kwargs)
