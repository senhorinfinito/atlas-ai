import os
import unittest
import json
from PIL import Image
import numpy as np

import lance
import pyarrow as pa

from atlas.data_sinks import sink


class VisionLanguageSinkTest(unittest.TestCase):
    def setUp(self):
        self.vl_path = "test_vl.jsonl"
        self.lance_path = "test_vl.lance"
        self.image_dir = "test_vl_images"
        os.makedirs(self.image_dir, exist_ok=True)

        # Create dummy image files
        self.image_paths = []
        for i in range(2):
            path = os.path.join(self.image_dir, f"image{i}.png")
            self.image_paths.append(path)
            Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)).save(path)

        with open(self.vl_path, "w") as f:
            f.write(json.dumps({"image": self.image_paths[0], "text": "text1"}) + "\n")
            f.write(json.dumps({"image": self.image_paths[1], "text": "text2"}) + "\n")

    def tearDown(self):
        os.remove(self.vl_path)
        if os.path.exists(self.lance_path):
            import shutil
            shutil.rmtree(self.lance_path)
        if os.path.exists(self.image_dir):
            import shutil
            shutil.rmtree(self.image_dir)

    def test_sink_vision_language(self):
        sink(self.vl_path, self.lance_path)
        dataset = lance.dataset(self.lance_path)
        self.assertEqual(dataset.count_rows(), 2)
        table = dataset.to_table()
        self.assertEqual(table.column_names, ["image", "text"])
        self.assertEqual(table.column("text").to_pylist(), ["text1", "text2"])

        images_data = []
        for path in self.image_paths:
            with open(path, "rb") as f:
                images_data.append(f.read())
        self.assertEqual(table.column("image").to_pylist(), images_data)


if __name__ == "__main__":
    unittest.main()
