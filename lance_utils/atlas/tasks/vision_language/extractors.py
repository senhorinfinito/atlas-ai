# lance_utils/tasks/vision_language/extractors.py
import json
from pathlib import Path
from typing import List, Dict, Any
import pyarrow as pa
from ..base import Extractor, Annotation

class ImageCaption(Annotation):
    caption: str

    @classmethod
    def arrow_field(cls) -> pa.Field:
        return pa.field("caption", pa.string())

    def to_dict(self) -> Dict[str, Any]:
        return {"caption": self.caption}

class JSONLCaptionExtractor(Extractor):
    """
    Expects a JSONL file where each line has "image_filename" and "caption" keys.
    Images are expected to be in an "images" subdirectory.
    """
    def extract(self, input_path: Path) -> Dict[str, List[Annotation]]:
        output = {}
        for jsonl_file in input_path.glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    image_filename = data.get("image_filename")
                    caption = data.get("caption")
                    if image_filename and caption:
                        output.setdefault(image_filename, []).append(
                            ImageCaption(caption=caption)
                        )
        return output
