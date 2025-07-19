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
from typing import Any, Dict, Optional, Union

import lance
import pyarrow as pa

from atlas.tasks.data_model.base import BaseDataset
from atlas.tasks.data_model.factory import create_dataset
from datasets import Dataset


def sink(
    data: Union[str, BaseDataset, Dataset],
    uri: Optional[str] = None,
    task: Optional[str] = None,
    format: Optional[str] = None,
    mode: str = "overwrite",
    **kwargs,
):
    """
    Sinks data from a given source to a specified destination in Lance format.

    This function provides a high-level API for converting and sinking data from various
    sources into a Lance dataset. The source can be a file path (e.g., a CSV file, a
    COCO annotation file), a `BaseDataset` object, or a Hugging Face `Dataset` object.

    Args:
        data (Union[str, BaseDataset, Dataset]): The data to be sunk.
        uri (str): The destination URI where the Lance dataset will be created.
        task (Optional[str], optional): The task for which the data is being sunk.
            For example, `object_detection`, `image_classification`, etc.
        format (Optional[str], optional): The format of the data. For example, `coco`,
            `yolo`, `csv`, etc.
        mode (str, optional): The mode for writing the data. Defaults to "overwrite".
        **kwargs: Additional options for the sink operation.
    """
    if isinstance(data, Dataset) and task is None:
        task = "hf"

    if isinstance(data, str):
        if uri is None:
            uri = f"{os.path.splitext(data)[0]}.lance"
        dataset = create_dataset(data, task=task, format=format, **kwargs)
    elif isinstance(data, Dataset) or (
        hasattr(data, "__iter__") and hasattr(data, "__next__")
    ):
        dataset = create_dataset(data, task=task, format=format, **kwargs)
    else:
        dataset = data

    dataset.to_lance(uri, mode=mode, **kwargs)
