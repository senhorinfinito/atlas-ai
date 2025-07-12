# lance_utils/tasks/object_detection/extractors.py
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
import pyarrow as pa
from PIL import Image
from ..base import Extractor, Annotation

class BoundingBox(Annotation):
    bbox: List[float]  # [x_min, y_min, width, height]
    category_id: int

    @classmethod
    def arrow_field(cls) -> pa.Field:
        return pa.field("bounding_boxes", pa.list_(pa.struct([
            pa.field("bbox", pa.list_(pa.float32(), 4)),
            pa.field("category_id", pa.int64())
        ])))

    def to_dict(self) -> Dict[str, Any]:
        return {"bbox": self.bbox, "category_id": self.category_id}

class COCOBoundingBoxExtractor(Extractor):
    def extract(self, input_path: Path) -> Dict[str, List[Annotation]]:
        annotations_file = input_path / "annotations.json"
        if not annotations_file.exists():
            return {}

        with open(annotations_file, "r") as f:
            coco_data = json.load(f)

        img_id_to_filename = {img['id']: img['file_name'] for img in coco_data.get('images', [])}
        
        output = {}
        for ann in coco_data.get('annotations', []):
            if 'bbox' in ann:
                img_id = ann['image_id']
                filename = img_id_to_filename.get(img_id)
                if filename:
                    output.setdefault(filename, []).append(
                        BoundingBox(bbox=ann['bbox'], category_id=ann['category_id'])
                    )
        return output

class YOLOBoundingBoxExtractor(Extractor):
    def extract(self, input_path: Path) -> Dict[str, List[Annotation]]:
        labels_dir = input_path / "labels"
        images_dir = input_path / "images"
        if not labels_dir.exists() or not images_dir.exists():
            return {}

        output = {}
        for label_file in labels_dir.glob("*.txt"):
            image_filename = f"{label_file.stem}.png" # Assume png, can be improved
            if not (images_dir / image_filename).exists():
                 image_filename = f"{label_file.stem}.jpg"
                 if not (images_dir / image_filename).exists():
                     continue

            annotations = []
            with Image.open(images_dir / image_filename) as img:
                img_w, img_h = img.size

            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    
                    abs_w = w * img_w
                    abs_h = h * img_h
                    x_min = (x_center * img_w) - (abs_w / 2)
                    y_min = (y_center * img_h) - (abs_h / 2)
                    
                    annotations.append(BoundingBox(
                        bbox=[x_min, y_min, abs_w, abs_h],
                        category_id=class_id
                    ))
            output[image_filename] = annotations
        return output
