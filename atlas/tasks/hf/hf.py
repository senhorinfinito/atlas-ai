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
from typing import Generator

import pyarrow as pa
from datasets import Dataset
from datasets.features.features import ClassLabel, Value, Sequence, Image, Audio

from atlas.tasks.data_model.base import BaseDataset
from atlas.utils.system import check_ffmpeg


class HFDataset(BaseDataset):
    """
    A dataset that wraps a Hugging Face dataset.
    """

    def __init__(self, data: Dataset, expand_level: int = 0, handle_nested_nulls: bool = False):
        super().__init__(data)
        self.expand_level = 1 if expand_level > 0 else 0
        self.handle_nested_nulls = handle_nested_nulls
        self._expansion_map = {}
        self.metadata.decode_meta = self._get_decode_meta()
        if any(isinstance(f, Audio) for f in self.data.features.values()):
            check_ffmpeg()

    def _get_decode_meta(self):
        decode_meta = {}
        for name, feature in self.data.features.items():
            if isinstance(feature, (Image, Audio)):
                decode_meta[name] = str(feature)
        return decode_meta

    def to_arrow_schema(self) -> pa.Schema:
        fields = []
        self._expansion_map.clear()
        for name, feature in self.data.features.items():
            fields.extend(self._convert_feature_to_arrow_fields(name, feature, self.expand_level))
        return pa.schema(fields)

    def _convert_feature_to_arrow_fields(self, name: str, feature, expand_level: int) -> list[pa.Field]:
        """
        _convert_feature_to_arrow_fields and _convert_feature_to_arrow_field work together to recursively
        define the schema. The former handles expansion of nested features, while the latter handles the
        conversion of individual features to Arrow fields.
        """
        # Expand dicts
        if expand_level > 0 and isinstance(feature, dict):
            fields = []
            # Sort keys to ensure consistent field order
            sorted_items = sorted(feature.items())
            for sub_name, sub_feature in sorted_items:
                new_name = f"{name}_{sub_name}"
                self._expansion_map[new_name] = (name, sub_name, "dict")
                fields.extend(self._convert_feature_to_arrow_fields(new_name, sub_feature, expand_level - 1))
            return fields
        # Expand lists of dicts
        if expand_level > 0 and isinstance(feature, Sequence) and isinstance(feature.feature, dict):
            fields = []
            # Sort keys to ensure consistent field order
            sorted_items = sorted(feature.feature.items())
            for sub_name, sub_feature in sorted_items:
                new_name = f"{name}_{sub_name}"
                self._expansion_map[new_name] = (name, sub_name, "list_of_dicts")
                # The sub-feature is now a list of the original sub-feature's type
                list_of_sub_feature = Sequence(feature=sub_feature)
                fields.append(self._convert_feature_to_arrow_field(new_name, list_of_sub_feature))
            return fields
        else:
            return [self._convert_feature_to_arrow_field(name, feature)]

    def _convert_feature_to_arrow_field(self, name: str, feature) -> pa.Field:
        """
        Converts a single Hugging Face feature to a PyArrow field. This method is called recursively
        for nested features like Sequence, dict, and list to define their nested structure.
        """
        if isinstance(feature, (Image, Audio)):
            return pa.field(name, pa.large_binary(), metadata={"lance:encoding": "binary"})
        if isinstance(feature, ClassLabel):
            return pa.field(name, pa.string())
        if isinstance(feature, Value):
            return pa.field(name, feature.pa_type)
        # For a Sequence (list), recursively call this function to determine the type of the elements.
        if isinstance(feature, Sequence):
            return pa.field(name, pa.list_(self._convert_feature_to_arrow_field("item", feature.feature).type))
        # For a dict, recursively call this function for each value to create a struct.
        if isinstance(feature, dict):
            # Sort keys to ensure consistent field order
            sorted_items = sorted(feature.items())
            return pa.field(name, pa.struct([self._convert_feature_to_arrow_field(k, v) for k, v in sorted_items]))
        # For a list, recursively call this function on the first element to determine the list type.
        if isinstance(feature, list):
            # Sort keys if the elements are dicts to ensure consistent field order
            if feature and isinstance(feature[0], dict):
                 # This path is tricky, assuming all dicts in list have same structure
                 # and sorting keys of first element is representative
                sorted_items = sorted(feature[0].items())
                struct_type = pa.struct([self._convert_feature_to_arrow_field(k, v) for k, v in sorted_items])
                return pa.field(name, pa.list_(struct_type))
            return pa.field(name, pa.list_(self._convert_feature_to_arrow_field("item", feature[0]).type))
        raise ValueError(f"Unsupported feature type for column '{name}': {feature}")

    def _process_column(self, column_data: pa.Array, feature) -> pa.Array:
        if column_data is None:
            return column_data

        if isinstance(feature, Image):
            serialized_data = []
            for img_data in column_data.to_pylist():
                if not img_data:
                    serialized_data.append(None)
                    continue
                if 'bytes' in img_data and img_data['bytes']:
                    serialized_data.append(img_data['bytes'])
                elif 'path' in img_data and img_data['path']:
                    with open(img_data['path'], 'rb') as f:
                        serialized_data.append(f.read())
                else:
                    buf = io.BytesIO()
                    img_data.save(buf, format='PNG')
                    serialized_data.append(buf.getvalue())
            return pa.array(serialized_data, type=pa.large_binary())

        if isinstance(feature, Audio):
            serialized_data = []
            for item in column_data.to_pylist():
                if item and 'path' in item and item['path']:
                    with open(item['path'], 'rb') as f:
                        serialized_data.append(f.read())
                elif item and 'bytes' in item:
                    serialized_data.append(item['bytes'])
                else:
                    serialized_data.append(None)
            return pa.array(serialized_data, type=pa.large_binary())

        if isinstance(feature, ClassLabel):
            return pa.array([feature.int2str(label) if label is not None else None for label in column_data.to_pylist()], type=pa.string())

        if isinstance(feature, Sequence):
            if isinstance(feature.feature, ClassLabel):
                return pa.array([[feature.feature.int2str(l) for l in label_list] if label_list is not None else None for label_list in column_data.to_pylist()])
            # This block handles a special case where a column is a list of dictionaries (Sequence of dicts),
            # and some of the values within those dictionaries are ClassLabels. The default Arrow conversion
            # does not handle converting the nested ClassLabels to strings, so we must do it manually.
            # This is not the most performant approach as it iterates in Python, but it correctly handles the conversion.
            if isinstance(feature.feature, dict):
                processed_list = []
                for list_of_dicts in column_data.to_pylist():
                    if list_of_dicts is None:
                        processed_list.append(None)
                        continue
                    new_list = []
                    for item_dict in list_of_dicts:
                        new_dict = {}
                        sorted_items = sorted(item_dict.items())
                        for k, v in sorted_items:
                            sub_feature = feature.feature[k]
                            if isinstance(sub_feature, ClassLabel) and v is not None:
                                new_dict[k] = sub_feature.int2str(v)
                            else:
                                new_dict[k] = v
                        new_list.append(new_dict)
                    processed_list.append(new_list)
                return pa.array(processed_list)

        return column_data

    def to_batches(self, batch_size: int = 1024, **kwargs) -> Generator[pa.RecordBatch, None, None]:
        """
        Yields batches of the dataset as Arrow RecordBatches.
        The logic iterates through the fields of the *target schema* and constructs each column
        one by one. This approach is robust because it guarantees that the output will match the
        schema, and it cleanly separates the logic for handling expanded and non-expanded columns.
        """
        schema = self.to_arrow_schema()

        if self.handle_nested_nulls:
            # Slower, but robust path that handles missing keys and null values in nested dicts.
            for batch in self.data.iter(batch_size=batch_size):
                arrays = []
                for field in schema:
                    if field.name in self._expansion_map:
                        original_name, sub_name, expansion_type = self._expansion_map[field.name]
                        original_feature = self.data.features[original_name]

                        if expansion_type == "dict":
                            sub_feature = original_feature[sub_name]
                            column_data = [row.get(sub_name) if row else None for row in batch[original_name]]
                            processed_data = self._process_column(pa.array(column_data), sub_feature)
                            arrays.append(pa.array(processed_data, type=field.type))
                            continue
                        elif expansion_type == "list_of_dicts":
                            sub_feature = original_feature.feature[sub_name]
                            processed_data = []
                            for list_of_dicts in batch[original_name]:
                                if list_of_dicts is None:
                                    processed_data.append(None)
                                    continue
                                sub_list = [d.get(sub_name) if d else None for d in list_of_dicts]
                                processed_data.append(sub_list)
                            
                            # Manually handle ClassLabel in list of dicts after expansion
                            if isinstance(sub_feature, ClassLabel):
                                processed_data = [[sub_feature.int2str(v) if v is not None else None for v in sub_list] if sub_list is not None else None for sub_list in processed_data]

                            arrays.append(pa.array(processed_data, type=field.type))
                            continue

                    column_data = pa.array(batch[field.name])
                    feature = self.data.features[field.name]
                    processed_data = self._process_column(column_data, feature)
                    arrays.append(pa.array(processed_data, type=field.type))

                yield pa.RecordBatch.from_arrays(arrays, schema=schema)
        else:
            # Faster, Arrow-native path. This is less robust to missing nested data.
            for batch in self.data.with_format("arrow").iter(batch_size=batch_size):
                arrays = []
                for field in schema:
                    if field.name in self._expansion_map:
                        original_name, sub_name, expansion_type = self._expansion_map[field.name]
                        original_feature = self.data.features[original_name]

                        if expansion_type == "dict":
                            column_data = batch[original_name]
                            if isinstance(column_data, pa.ChunkedArray):
                                column_data = column_data.combine_chunks()
                            
                            sub_column_data = column_data.field(sub_name)
                            sub_feature = original_feature[sub_name]
                            
                            processed_data = self._process_column(sub_column_data, sub_feature)
                            arrays.append(pa.array(processed_data, type=field.type))
                            continue
                        elif expansion_type == "list_of_dicts":
                            column_data = batch[original_name]
                            if isinstance(column_data, pa.ChunkedArray):
                                column_data = column_data.combine_chunks()

                            # Extract sub-list from list of structs
                            sub_lists = []
                            for struct_list in column_data.to_pylist():
                                if struct_list is None:
                                    sub_lists.append(None)
                                    continue
                                sub_list = [d[sub_name] if d and sub_name in d else None for d in struct_list]
                                sub_lists.append(sub_list)

                            sub_feature = original_feature.feature[sub_name]
                            processed_data = self._process_column(pa.array(sub_lists), Sequence(feature=sub_feature))
                            arrays.append(pa.array(processed_data, type=field.type))
                            continue

                    column_data = batch[field.name]
                    feature = self.data.features[field.name]
                    processed_data = self._process_column(column_data, feature)
                    arrays.append(pa.array(processed_data, type=field.type))

                yield pa.RecordBatch.from_arrays(arrays, schema=schema)

    @property
    def schema(self) -> pa.Schema:
        return self.to_arrow_schema()