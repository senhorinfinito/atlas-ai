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
from typing import Any, Union, Optional, Tuple

import pyarrow.parquet as pq
import pyarrow as pa

from atlas.tasks.data_model.base import BaseDataset


def infer_dataset_type(data: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Infers the dataset type (task and format) based on the data source.

    Args:
        data (str): The data source (file path or directory).

    Returns:
        A tuple containing the inferred task and format.
    """
    if os.path.isdir(data):
        if os.path.exists(os.path.join(data, "annotations")) and any(
            f.endswith(".json") for f in os.listdir(os.path.join(data, "annotations"))
        ):
            if "instances" in str(os.listdir(os.path.join(data, "annotations"))):
                return "object_detection", "coco"
            elif "segmentation" in str(os.listdir(os.path.join(data, "annotations"))):
                return "segmentation", "coco"
        elif os.path.exists(os.path.join(data, "images")) and os.path.exists(
            os.path.join(data, "labels")
        ):
            return "object_detection", "yolo"
    elif data.endswith(".csv"):
        return "tabular", "csv"
    elif data.endswith(".parquet"):
        return "tabular", "parquet"
    elif data.endswith(".json"):
        # A json file could be a COCO dataset
        return "object_detection", "coco"
    elif data.endswith(".txt"):
        return "text", "text"
    elif data.endswith(".jsonl"):
        with open(data, "r") as f:
            first_line = f.readline()
            try:
                record = json.loads(first_line)
                if "query" in record and "documents" in record:
                    return "ranking", "ranking"
                elif "instruction" in record and "output" in record:
                    return "instruction", "instruction"
                elif "image" in record and "text" in record:
                    return "vision_language", "vision_language"
                elif "question" in record and "thought" in record and "answer" in record:
                    return "cot", "cot"
                elif "sentence1" in record and "sentence2" in record and "label" in record:
                    return "paired_text", "paired_text"
                elif "sentence1" in record and "sentence2" in record and "similarity_score" in record:
                    return "similarity", "similarity"
            except json.JSONDecodeError:
                pass  # Not a valid jsonl file

    return None, None


def create_dataset(
    data: Union[str, Any],
    task: Optional[str] = None,
    format: Optional[str] = None,
    **kwargs,
) -> BaseDataset:
    """
    Factory function to create a dataset object based on the given options.

    Args:
        data (str): The data source.
        task (Optional[str], optional): The task for which the data is being sunk.
        format (Optional[str], optional): The format of the data.
        **kwargs: Additional options for creating the dataset.

    Returns:
        BaseDataset: A dataset object.
    """
    if not isinstance(data, str):  # Hugging Face dataset
        if task == "instruction":
            from atlas.tasks.instruction.instruction import InstructionDataset

            return InstructionDataset(data)
        elif task == "ranking":
            from atlas.tasks.ranking.ranking import RankingDataset

            return RankingDataset(data)
        elif task == "paired_text":
            from atlas.tasks.paired_text.paired_text import PairedTextDataset

            return PairedTextDataset(data)
        elif task == "similarity":
            from atlas.tasks.similarity.similarity import SimilarityDataset

            return SimilarityDataset(data)
        elif task == "cot":
            from atlas.tasks.cot.cot import CoTDataset

            return CoTDataset(data)
        elif task == "hf":
            from atlas.tasks.hf.hf import HFDataset

            return HFDataset(data, **kwargs)

    if not task or not format:
        inferred_task, inferred_format = infer_dataset_type(data)
        task = task or inferred_task
        format = format or inferred_format

    if task == "object_detection":
        if format == "coco":
            from atlas.tasks.object_detection.coco import CocoDataset

            return CocoDataset(data, **kwargs)
        elif format == "yolo":
            from atlas.tasks.object_detection.yolo import YoloDataset

            return YoloDataset(data, **kwargs)
    elif task == "segmentation":
        if format == "coco":
            from atlas.tasks.segmentation.coco import CocoSegmentationDataset

            return CocoSegmentationDataset(data, **kwargs)
    elif task == "tabular":
        if format == "csv":
            from atlas.tasks.tabular.csv import CsvDataset

            return CsvDataset(data)
        elif format == "parquet":
            from atlas.tasks.tabular.parquet import ParquetDataset

            return ParquetDataset(data)
    elif task == "text":
        if format == "text":
            from atlas.tasks.text.text import TextDataset

            return TextDataset(data)
    elif task == "instruction":
        if format == "instruction":
            from atlas.tasks.instruction.instruction import InstructionDataset

            return InstructionDataset(data)
    elif task == "ranking":
        if format == "ranking":
            from atlas.tasks.ranking.ranking import RankingDataset

            return RankingDataset(data)
    elif task == "vision_language":
        if format == "vision_language":
            from atlas.tasks.vision_language.vision_language import (
                VisionLanguageDataset,
            )

            return VisionLanguageDataset(data)
    elif task == "cot":
        if format == "cot":
            from atlas.tasks.cot.cot import CoTDataset

            return CoTDataset(data)
    elif task == "paired_text":
        if format == "paired_text":
            from atlas.tasks.paired_text.paired_text import PairedTextDataset

            return PairedTextDataset(data)
    elif task == "similarity":
        if format == "similarity":
            from atlas.tasks.similarity.similarity import SimilarityDataset

            return SimilarityDataset(data)

    raise ValueError(f"Unsupported data format or task: {data}, {task}, {format}")
