# lance_utils/tasks/classification/extractors.py
from pathlib import Path
from typing import List, Dict, Any
import pyarrow as pa
from ..base import Extractor, Annotation

class Classification(Annotation):
    class_label: str

    @classmethod
    def arrow_field(cls) -> pa.Field:
        return pa.field("class_label", pa.string())

    def to_dict(self) -> Dict[str, Any]:
        return {"class_label": self.class_label}

class FolderNameClassificationExtractor(Extractor):
    def extract(self, input_path: Path) -> Dict[str, List[Annotation]]:
        output = {}
        for class_dir in input_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for image_file in class_dir.glob("*"):
                    if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        output[image_file.name] = [Classification(class_label=class_name)]
        return output
