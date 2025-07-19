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

from typing import Generator, Union

import pyarrow as pa
from datasets import Dataset, Features
from datasets.features.features import ClassLabel, Value, Sequence

from atlas.tasks.data_model.base import BaseDataset


class HFDataset(BaseDataset):
    """
    A dataset that wraps a Hugging Face dataset.
    """

    def __init__(self, data: Dataset):
        super().__init__(data)
        self._validate_schema()

    def _validate_schema(self):
        """
        Validates the schema of the Hugging Face dataset.
        For now, it only supports text-based columns.
        """
        for name, feature in self.data.features.items():
            if not self._is_supported_feature(feature):
                raise ValueError(
                    f"Unsupported feature type for column '{name}': {feature}. "
                    f"Currently, only text-based features are supported."
                )

    def _is_supported_feature(self, feature) -> bool:
        """
        Checks if a feature is supported.
        """
        if isinstance(feature, ClassLabel):
            return True
        if isinstance(feature, Value):
            return pa.types.is_string(feature.pa_type) or pa.types.is_binary(feature.pa_type) or pa.types.is_integer(feature.pa_type) or pa.types.is_floating(feature.pa_type)
        if isinstance(feature, Sequence):
            return self._is_supported_feature(feature.feature)
        if isinstance(feature, dict):
            return all(self._is_supported_feature(f) for f in feature.values())
        if isinstance(feature, list):
            return all(self._is_supported_feature(f) for f in feature)
        return False

    def to_batches(self, batch_size: int = 1024, **kwargs) -> Generator[pa.RecordBatch, None, None]:
        """
        Yields batches of the dataset as Arrow RecordBatches.
        """
        schema = self.schema
        for batch in self.data.iter(batch_size=batch_size):
            arrays = [pa.array(batch[name], type=schema.field(name).type) for name in schema.names]
            yield pa.RecordBatch.from_arrays(arrays, schema=schema)

    @property
    def schema(self) -> pa.Schema:
        """
        Returns the schema of the dataset.
        """
        return self.data.features.arrow_schema
