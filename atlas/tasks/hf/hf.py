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

    def __init__(self, data: Dataset, expand_level: int = 0):
        super().__init__(data)
        self.expand_level = expand_level
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
            fields.extend(self._convert_feature_to_arrow_fields(name, feature, self.expand_level))
        return pa.schema(fields)

    def _convert_feature_to_arrow_fields(self, name: str, feature, expand_level: int) -> list[pa.Field]:
        """
        Converts a Hugging Face feature to a list of PyArrow fields.
        """
        if expand_level > 0 and isinstance(feature, (dict, Sequence)):
            if isinstance(feature, dict):
                sub_features = feature.items()
            else: # Sequence
                if isinstance(feature.feature, dict):
                    sub_features = feature.feature.items()
                else:
                    return [self._convert_feature_to_arrow_field(name, feature)]

            fields = []
            for sub_name, sub_feature in sub_features:
                fields.extend(self._convert_feature_to_arrow_fields(f"{name}_{sub_name}", sub_feature, expand_level - 1))
            return fields
        else:
            return [self._convert_feature_to_arrow_field(name, feature)]

    def _convert_feature_to_arrow_field(self, name: str, feature) -> pa.Field:
        """
        Converts a Hugging Face feature to a PyArrow field.
        """
        if isinstance(feature, (Image, Audio)):
            return pa.field(name, pa.large_binary(), metadata={"lance:encoding": "binary"})
        if isinstance(feature, ClassLabel):
            return pa.field(name, pa.string())
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
            for field in schema:
                name = field.name
                feature = self.data.features
                # Handle expanded columns
                if "_" in name and name.split("_")[0] in self.data.features and isinstance(self.data.features[name.split("_")[0]], (dict, Sequence)):
                    name_parts = name.split("_")
                    column_data = batch[name_parts[0]]
                    if isinstance(column_data, pa.ChunkedArray):
                        column_data = column_data.combine_chunks()
                    feature = feature[name_parts[0]]
                    for part in name_parts[1:]:
                        if column_data is None:
                            break
                        column_data = column_data.field(part)
                        if isinstance(feature, Sequence):
                            feature = feature.feature
                        else:
                            feature = feature[part]
                else:
                    column_data = batch[name]
                    feature = feature[name]

                if isinstance(feature, Image):
                    serialized_data = []
                    for img_data in column_data.to_pylist():
                        if img_data:
                            if 'bytes' in img_data and img_data['bytes']:
                                serialized_data.append(img_data['bytes'])
                            elif 'path' in img_data and img_data['path']:
                                with open(img_data['path'], 'rb') as f:
                                    serialized_data.append(f.read())
                            else:
                                # Handle PIL Image objects
                                buf = io.BytesIO()
                                img_data.save(buf, format='PNG')
                                serialized_data.append(buf.getvalue())
                        else:
                            serialized_data.append(None)
                    arrays.append(pa.array(serialized_data, type=pa.large_binary()))
                elif isinstance(feature, Audio):
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
                elif isinstance(feature, Sequence) and isinstance(feature.feature, dict):
                    # Handle list of dicts by converting ClassLabels to strings
                    converted_data = []
                    for list_of_dicts in column_data.to_pylist():
                        if list_of_dicts is None:
                            converted_data.append(None)
                            continue
                        new_list = []
                        for item_dict in list_of_dicts:
                            new_dict = item_dict.copy()
                            for key, value in item_dict.items():
                                sub_feature = feature.feature[key]
                                if isinstance(sub_feature, ClassLabel):
                                    new_dict[key] = sub_feature.int2str(value) if value is not None else None
                                elif isinstance(sub_feature, Sequence) and isinstance(sub_feature.feature, ClassLabel):
                                     new_dict[key] = [sub_feature.feature.int2str(v) for v in value] if value is not None else None
                            new_list.append(new_dict)
                        converted_data.append(new_list)
                    arrays.append(pa.array(converted_data, type=field.type))
                elif isinstance(feature, ClassLabel):
                    string_labels = [feature.int2str(label) if label is not None else None for label in column_data.to_pylist()]
                    arrays.append(pa.array(string_labels, type=pa.string()))
                else:
                    arrays.append(pa.array(column_data, type=field.type))
            yield pa.RecordBatch.from_arrays(arrays, schema=schema)

    @property
    def schema(self) -> pa.Schema:
        """
        Returns the schema of the dataset.
        """
        return self.to_arrow_schema()
