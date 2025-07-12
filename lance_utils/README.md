# Lance Utils

A flexible, annotation-first library for discovering, parsing, and converting complex local datasets into the high-performance [Lance format](https://lancedb.github.io/lance/).

This library moves beyond rigid, task-specific converters. Instead of forcing your data into a predefined schema, `lance_utils` automatically discovers all supported annotations in your dataset and dynamically builds a schema to match.

If your dataset has bounding boxes, preference data for RLHF, and image captions all mixed together, `lance_utils` will find them all and create a single, unified Lance dataset with a column for each annotation type.

## Key Features

- **Annotation-First Design:** Automatically discovers and parses multiple annotation types from a single data source.
- **Dynamic Schema Generation:** The Lance dataset schema is built on the fly based on the annotations found.
- **Extensible:** Easily add support for new data formats and annotation types by creating simple `Extractor` plugins.
- **Resource-Aware:** Automatically determines a safe batch size based on your system's available memory to prevent out-of-memory errors.
- **Simple CLI & Python API:** Use the library from the command line or directly in your Python scripts.

## Installation

```bash
# Navigate to the lance_utils directory
pip install .
```

## Usage

### CLI Usage

The primary command is `lance_utils build`. Just point it at your data directory, and it will auto-detect and convert everything it understands.

```bash
lance_utils build -i /path/to/your/dataset -o my_dataset.lance
```

**Other Commands:**

- **Inspect a dataset's schema:**
  ```bash
  lance_utils inspect -i my_dataset.lance
  ```
- **Visualize a sample of the dataset (for image-based datasets):**
  ```bash
  lance_utils visualize -i my_dataset.lance
  ```

### Python API Usage

You can use `lance_utils` directly in your Python code for more programmatic control.

```python
import lance_utils

# Path to your directory with mixed annotations
source_data = "/path/to/your/dataset"
output_lance_dataset = "my_dataset.lance"

# Build the dataset
atlas.build(
    input_path=source_data,
    output_path=output_lance_dataset
)

print("Dataset conversion complete!")
```

## Examples

The [`examples/`](./examples) directory contains scripts that demonstrate how to download and convert several common public datasets.

To run an example, navigate to the `examples` directory and run the script directly:
```bash
cd examples
python convert_object_detection.py
```

## Supported Formats (Auto-Detected)

The `build` command will automatically search for and parse any of the following formats found in the input directory:

- **Plain Text (`.txt`):** For LLM pre-training. Each `.txt` file is treated as a separate document.
- **Preference Data (`.jsonl`):** For DPO/RLHF. Each line should be a JSON object with `"prompt"`, `"chosen"`, and `"rejected"` keys.
- **COCO Bounding Boxes (`annotations.json`):** For object detection.
- **YOLO Bounding Boxes (`labels/*.txt` and `data.yaml`):** For object detection.
- **Folder-based Image Classification:** Images are classified based on the name of their parent directory.
- **Image-Caption Pairs (`captions.jsonl`):** For VLM training. Each line should be a JSON object with `"image_filename"` and `"caption"` keys.

## The Annotation-First Philosophy

This library is built on the idea that a dataset is a collection of **data items** (like images or text files) and **annotations** (like bounding boxes, captions, or preference pairs).

Instead of forcing a dataset into a rigid "task" like "object detection", `lance_utils` inspects the source directory with a suite of small, specialized **Extractors**. Each extractor knows how to find one type of annotation (e.g., the `YOLOBoundingBoxExtractor` looks for YOLO-formatted labels).

The `LanceBuilder` collects all the annotations found by all the extractors and dynamically constructs a Lance dataset with a schema that perfectly fits the data. This makes the library incredibly flexible and easy to extend.

## Contributing

To add support for a new data format or annotation type, you simply need to create a new `Annotation` and `Extractor` class. See the `DESIGN.md` file for a technical overview.