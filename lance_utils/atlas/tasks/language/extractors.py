# lance_utils/tasks/language/extractors.py
import json
from pathlib import Path
from typing import List, Dict
from ..base import Extractor, Annotation, RawText, PreferenceData

class PlainTextExtractor(Extractor):
    """Extracts raw text from .txt files for pre-training."""
    def extract(self, input_path: Path) -> Dict[str, List[Annotation]]:
        output = {}
        for txt_file in input_path.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
                output[txt_file.name] = [RawText(text=content)]
        return output

class DPOPreferenceExtractor(Extractor):
    """Extracts preference data from a JSONL file for DPO."""
    def extract(self, input_path: Path) -> Dict[str, List[Annotation]]:
        output = {}
        for jsonl_file in input_path.glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    if "prompt" in data and "chosen" in data and "rejected" in data:
                        item_id = f"{jsonl_file.stem}_{i}"
                        output[item_id] = [PreferenceData(
                            prompt=data["prompt"],
                            chosen=data["chosen"],
                            rejected=data["rejected"]
                        )]
        return output