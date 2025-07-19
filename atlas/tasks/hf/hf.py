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

import io
from typing import Generator, Union

import pyarrow as pa
from datasets import Dataset, Features
from datasets.features.features import ClassLabel, Value, Sequence, Image, Audio

from atlas.tasks.data_model.base import BaseDataset
from atlas.utils.system import check_ffmpeg


class HFDataset(BaseDataset):
    """
    A dataset that wraps a Hugging Face dataset.
    """

    def __init__(self, data: Dataset):
        super().__init__(data)
        self.metadata.decode_meta = self._get_decode_meta()
        if any(isinstance(f, Audio) for f in self.data.features.values()):
            check_ffmpeg()

    def _get_decode_meta(self):
        """
        Generates the decode metadata for the dataset.
        """
        decode_meta = {}
        for name, feature in self.data.features.items():
            if isinstance(feature, (Image, Audio)):
                decode_meta[name] = str(feature)
        return decode_meta

    def to_arrow_schema(self) -> pa.Schema:
        """
        Returns the schema of the dataset.
        """
        fields = []
        for name, feature in self.data.features.items():
            fields.append(self._convert_feature_to_arrow_field(name, feature))
        return pa.schema(fields)

    def _convert_feature_to_arrow_field(self, name: str, feature) -> pa.Field:
        """
        Converts a Hugging Face feature to a PyArrow field.
        """
        if isinstance(feature, (Image, Audio)):
            return pa.field(name, pa.large_binary(), metadata={"lance:encoding": "binary"})
        if isinstance(feature, ClassLabel):
            return pa.field(name, feature.names)
        if isinstance(feature, Value):
            return pa.field(name, feature.pa_type)
        if isinstance(feature, Sequence):
            return pa.field(name, pa.list_(self._convert_feature_to_arrow_field(name, feature.feature).type))
        if isinstance(feature, dict):
            return pa.field(name, pa.struct([self._convert_feature_to_arrow_field(k, v) for k, v in feature.items()]))
        if isinstance(feature, list):
            return pa.field(name, pa.list_(self._convert_feature_to_arrow_field(name, feature[0]).type))
        raise ValueError(f"Unsupported feature type for column '{name}': {feature}")

    def to_batches(self, batch_size: int = 1024, **kwargs) -> Generator[pa.RecordBatch, None, None]:
        """
        Yields batches of the dataset as Arrow RecordBatches.
        """
        schema = self.to_arrow_schema()
        for batch in self.data.with_format("arrow").iter(batch_size=batch_size):
            arrays = []
            for name in schema.names:
                feature = self.data.features[name]
                column_data = batch[name]
                if isinstance(feature, (Image, Audio)):
                    serialized_data = []
                    for item in column_data.to_pylist():
                        if item and 'path' in item and item['path']:
                            with open(item['path'], 'rb') as f:
                                serialized_data.append(f.read())
                        elif item and 'bytes' in item:
                            serialized_data.append(item['bytes'])
                        else:
                            serialized_data.append(None)
                    arrays.append(pa.array(serialized_data, type=pa.large_binary()))
                else:
                    arrays.append(pa.array(column_data, type=schema.field(name).type))
            yield pa.RecordBatch.from_arrays(arrays, schema=schema)

    @property
    def schema(self) -> pa.Schema:
        """
        Returns the schema of the dataset.
        """
        return self.to_arrow_schema()
