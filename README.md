# Atlas

<p align="center">
  <img src="https://storage.googleapis.com/atlas-resources/logo.png" alt="Atlas Logo" width="200"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/atlas-ai/">
    <img src="https://img.shields.io/pypi/v/atlas-ai.svg" alt="PyPI Version">
  </a>
  <a href="https://github.com/your-repo/atlas/actions">
    <img src="https://github.com/your-repo/atlas/workflows/CI/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/your-repo/atlas">
    <img src="https://codecov.io/gh/your-repo/atlas/branch/main/graph/badge.svg" alt="Codecov">
  </a>
</p>

Atlas is a data-centric AI framework for curating, indexing, and analyzing massive datasets for deep learning applications. It provides a suite of tools to streamline the entire data lifecycle, from initial data ingestion to model training and analysis.

## Core Operations

The vision for Atlas is to provide a comprehensive solution for managing large-scale datasets in AI development. The framework is built around three core operations:

-   **Sink:** Ingest data from any source and format into an optimized Lance dataset.
```
        +-----------------------+      +----------------------+      +---------------------+
        |   Raw Data Sources    |      |      Atlas Sink      |      |   Lance Dataset     |
        |  (COCO, YOLO, CSV)    |----->|  (Auto-detection,    |----->|  (Optimized Storage)|
        |                       |      |   Metadata Extraction)|      |                     |
        +-----------------------+      +----------------------+      +---------------------+
```

-   **Index:** Create powerful, multi-modal indexes (FTS/BM25, Vector embeddings, Hybrid, or custom features) on your data to enable fast and efficient search and retrieval.
```
        +---------------------+      +----------------------+      +-----------------------------+
        |   Lance Dataset     |      |       Index          |      |   Indexed Dataset           |
        | (Optimized Storage) |----->|  (Vector & Metadata  |----->| (Vector Search, SQL Filters)|
        | (Larger than memory)|      |      Indexing)       |      | (For massive datasets)      |
        +---------------------+      +----------------------+      +-----------------------------+
```

-   **Analyse:** Analyse your datasets to gain insights, identify patterns, and debug your models (Run EDA and filters of larger-than-memory datasets).
```
        +-----------------------+      +------------------------+      +----------------------+
        |   Indexed Dataset     |      |     Atlas Analyse      |      |   Insights &         |
        |    (Fast Queries)     |----->| (Embedding Analysis,   |----->|   Visualizations     |
        |                       |      |  Quality Checks, etc.) |      |                      |
        +-----------------------+      +------------------------+      +----------------------+
```
**Training** Connect with your desired trainer and directly train from your sink source without any transformation required.
```
        +-----------------------+      +------------------------+      +----------------------+
        |   Indexed Dataset     |      |      Trainer           |      |   Insights &         |
        |    (Fast Queries)     |----->| (PyTorch, TF, etc.)    |----->|   Models             |
        |                       |      |                        |      |                      |
        +-----------------------+      +------------------------+      +----------------------+
```

Here is an end-to-end example of how to `sink` a dataset and then `index` it.
```python
import atlas
from atlas.index import Indexer
from datasets import load_dataset

# --- 1. Sink a dataset from Hugging Face Hub ---
# We'll use a dataset with images and text captions.
dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
atlas.sink(dataset, "pokemon.lance")

# The sink operation creates a Lance dataset with the following structure:
# +------------------------------------+-----------------------------------------+
# | image                              | text                                    |
# +====================================+=========================================+
# | <PIL.PngImagePlugin.PngImageFile>  | a drawing of a pink pokemon with a...   |
# +------------------------------------+-----------------------------------------+
# | <PIL.PngImagePlugin.PngImageFile>  | a green and yellow pokemon with a...    |
# +------------------------------------+-----------------------------------------+


# --- 2. Initialize the Indexer ---
# The Indexer attaches to the dataset you just sinked.
idx = Indexer("pokemon.lance")

# --- 3. Create Indexes ---
# Create a vector index on the 'image' column.
# Atlas will automatically use a default model to generate embeddings.
idx.create_index(column="image", index_type="vector")

# Create a Full-Text Search (FTS) index on the 'text' column.
idx.create_index(column="text", index_type="fts")

# --- 4. List and verify indexes ---
# The 'vector' column is added for embeddings, and indexes are created.
idx.list_indexes()
# ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
# ┃ Column Name ┃ Data Type                         ┃ Index Type ┃
# ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⇇━━━━━━━━━━━━┩
# │ image       │ binary                            │ None       │
# │ text        │ string                            │ text_idx   │
# │ vector      │ fixed_size_list<item: float>[768] │ vector_idx │
# └─────────────┴───────────────────────────────────┴────────────┘
```
---

# Sink

The **Sink** operation allows you to ingest data from any source and format into an optimized [Lance](https://lancedb.github.io/lance/) dataset. Atlas automatically infers the dataset type, extracts rich metadata, and stores the data in a self-contained, portable format.

## Features

-   **Automatic Data Ingestion:** The `sink` command automatically detects the dataset type (e.g., COCO, YOLO, CSV) and infers the optimal way to ingest the data into Lance.
-   **Rich Metadata Extraction:** Atlas extracts a wide range of metadata from your datasets, including image dimensions, class names, captions, and keypoints.
-   **Self-Contained Datasets:** All data, including images and other binary assets, is stored directly in the Lance dataset, making it easy to share and version your data.
-   **Extensible Architecture:** The framework is designed to be easily extensible, allowing you to add support for new data formats, tasks, and indexing strategies.
-   **Command-Line and Python API:** Atlas provides both a simple and intuitive command-line interface and a powerful Python API for programmatic access.

## Installation

To use Atlas, you need to have FFmpeg installed on your system.

**macOS**
```bash
brew install ffmpeg
```

**Linux**
```bash
sudo apt-get install ffmpeg
```

If you have installed ffmpeg and are still seeing errors, you may need to set the `DYLD_LIBRARY_PATH` environment variable. For example, if you installed ffmpeg with homebrew, you can run:
```bash
export DYLD_LIBRARY_PATH=$(brew --prefix)/lib:$DYLD_LIBRARY_PATH
```

Then, install Atlas using pip:
```bash
pip install atlas-ai
```

For audio datasets, you will need to install the `soundfile` dependency. You can do this by running:
```bash
pip install atlas-ai[audio]
```


## Usage

### Python API (Recommended)

The `atlas` Python API provides a flexible and powerful way to sink your datasets.

**Sinking from Hugging Face Datasets**

This is the recommended way to use Atlas. You can sink a dataset directly from the Hugging Face Hub. Atlas will preserve the original schema and automatically handle multimodal data like images and audio.

Here's an example using the `lambdalabs/pokemon-blip-captions` dataset:

```python
from datasets import load_dataset
import atlas

# Load dataset from Hugging Face
dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")

# Sink the dataset to Lance format
atlas.sink(dataset, "pokemon.lance")
```

<p align="center">
  <img src="examples/data/coco_visualization.png" alt="COCO Visualization" width="500"/>
</p>

<details open>
<summary>Click to see sample data</summary>

**Sample Data:**
```
+------------------------------------+------------------+---------+----------+------------------------------------------------------------+----------------------------------------------------------+
| image                              | file_name        |   width |   height | label                                                      | bbox                                                     |
+====================================+==================+=========+==========+============================================================+==========================================================+
| b'\xff\xd8\xff\xe0\x00\x10JFIF'... | 000000397133.jpg |     640 |      427 | [44 67  1 49 51 51 79  1 47 47 51 51 56 50 56 56 79 57 81] | [array([217.62, 240.54,  38.99,  57.75], dtype=float32)  |
+------------------------------------+------------------+---------+----------+------------------------------------------------------------+----------------------------------------------------------+
```
</details>

<details>
<summary>Automatically expand nested schemas</summary>

For nested Hugging Face datasets, you can use the `expand_level` argument to flatten the structure. For example, `expand_level=1` will expand the first level of nested columns.

If your nested data contains missing keys or `None` values, the default expansion may produce incorrect results. To handle these cases gracefully, set `handle_nested_nulls=True`. This uses a more robust (but slightly slower) method to ensure nulls are preserved correctly.

**Example:**

Given a dataset with a nested column `nested`:
```python
from datasets import Dataset, Features, Value
import atlas

data = [
    {"nested": {"a": 1, "b": "one"}},
    {"nested": {"a": 2, "b": "two", "c": True}},
    {"nested": {"a": 3}},
    {},
]
features = Features({
    "nested": {
        "a": Value("int64"),
        "b": Value("string"),
        "c": Value("bool"),
    }
})
dataset = Dataset.from_list(data, features=features)

# Sink with expansion
atlas.sink(dataset, "expanded.lance", task="hf", expand_level=1, handle_nested_nulls=True)
```

**Original Data Table:**
```
+------------------------------------------+
| nested                                   |
+==========================================+
| {'a': 1, 'b': 'one', 'c': None}           |
+------------------------------------------+
| {'a': 2, 'b': 'two', 'c': True}           |
+------------------------------------------+
| {'a': 3, 'b': None, 'c': None}            |
+------------------------------------------+
| None                                     |
+------------------------------------------+
```

**Expanded Data Table (`expand_level=1`):**
```
+----------+----------+----------+
| nested_a | nested_b | nested_c |
+==========+==========+==========+
| 1        | 'one'    | None     |
+----------+----------+----------+
| 2        | 'two'    | True     |
+----------+----------+----------+
| 3        | None     | None     |
+----------+----------+----------+
| None     | None     | None     |
+----------+----------+----------+
```
</details>

<details>
<summary>Task-based or File-format based sinks are also supported</summary>

**Object Detection (COCO format)**
```python
import atlas
atlas.sink("examples/data/coco/annotations/instances_val2017_small.json")
```

**Object Detection (YOLO)**

```python
import atlas
atlas.sink("examples/data/yolo/coco128")
```

**Segmentation (COCO)**

```python
import atlas
atlas.sink("examples/data/coco/annotations/instances_val2017_small.json", task="segmentation")
```
**sink accepts optional `task` arg to determine the format of dataset. It's inferred if no provided**

**Tabular (CSV file format)**

```python
import atlas
atlas.sink("examples/data/dummy.csv")
```

**Text file format**

```python
import atlas
atlas.sink("examples/data/dummy.txt")
```
**Parquet file format**

```python
import atlas
atlas.sink("examples/data/dummy.parquet")
```

LLM based task types are also supported

**Instruction**

```python
import atlas
atlas.sink("examples/data/dummy.jsonl")
```

**Ranking**

```python
import atlas
atlas.sink("examples/data/dummy_ranking.jsonl")
```

**Vision-Language**

```python
import atlas
atlas.sink("examples/data/dummy_vl.jsonl")
```

**Chain of Thought**

```python
import atlas
atlas.sink("examples/data/dummy_cot.jsonl")
```

**Paired Text**

```python
import atlas
atlas.sink("examples/data/stsb_train.jsonl")
```
</details>

<details>
<summary>Extend the Sink API</summary>

You can also import specific task types and use them directly or even subclass them for more advanced use cases. For example, let's create a custom sink that adds an `image_url` to the COCO dataset.

```python
from atlas.tasks.object_detection.coco import CocoDataset
import pyarrow as pa
import atlas

class CocoDatasetWithImageURL(CocoDataset):
    def __init__(self, data: str, **kwargs):
        super().__init__(data, **kwargs)
        self.base_url = "http://images.cocodataset.org/val2017/"

    def to_batches(self, batch_size: int = 1024):
        for batch in super().to_batches(batch_size):
            file_names = batch.column("file_name").to_pylist()
            image_urls = [self.base_url + file_name for file_name in file_names]
            yield batch.add_column(0, pa.field("image_url", pa.string()), pa.array(image_urls, type=pa.string()))

# Usage
custom_coco_dataset = CocoDatasetWithImageURL(
    "examples/data/coco/annotations/instances_val2017_small.json",
    image_root="examples/data/coco/images"
)
atlas.sink(custom_coco_dataset, "coco_with_url.lance")
```

</details>

### CLI

The `atlas` CLI provides a simple way to interact with your datasets.

<details>
<summary>CLI Usage</summary>

<details>
<summary>Object Detection (COCO)</summary>

```bash
atlas sink examples/data/coco/annotations/instances_val2017_small.json
```

</details>

<details>
<summary>Object Detection (YOLO)</summary>

```bash
atlas sink examples/data/yolo/coco128
```
<p align="center">
  <img src="examples/data/yolo_visualization.png" alt="YOLO Visualization" width="500"/>
</p>

</details>

<details>
<summary>Segmentation (COCO)</summary>

```bash
atlas sink examples/data/coco/annotations/instances_val2017_small.json --task segmentation
```
<p align="center">
  <img src="examples/data/coco_segmentation_visualization.png" alt="COCO Segmentation Visualization" width="500"/>
</p>

</details>

<details>
<summary>Tabular (CSV)</summary>

```bash
atlas sink examples/data/dummy.csv
```

</details>

<details>
<summary>Text</summary>

```bash
atlas sink examples/data/dummy.txt
```

</details>

<details>
<summary>Instruction</summary>

```bash
atlas sink examples/data/dummy.jsonl
```

</details>

<details>
<summary>Embedding</summary>

```bash
atlas sink examples/data/dummy.parquet
```

</details>

<details>
<summary>Ranking</summary>

```bash
atlas sink examples/data/dummy_ranking.jsonl
```

</details>

<details>
<summary>Vision-Language</summary>

```bash
atlas sink examples/data/dummy_vl.jsonl
```

</details>

<details>
<summary>Chain of Thought</summary>

```bash
atlas sink examples/data/dummy_cot.jsonl
```

</details>

<details>
<summary>Paired Text</summary>

```bash
atlas sink examples/data/stsb_train.jsonl
```

</details>
</details>

---
# Index

The **Index** operation allows you to create powerful, multi-modal indexes on your sinked dataset. This enables fast and efficient search and retrieval, which is crucial for working with large-scale AI datasets.

## Features

-   **Multi-Modal Indexing:** Create vector embeddings for text, images, and other modalities, or generate traditional FTS (Full-Text Search) indexes.
-   **Automatic Vectorization:** If you create a vector index on a column with raw data (like text or images), Atlas will automatically generate embeddings using a default model.
-   **Flexible and Extensible:** The indexing framework is designed to be extensible, allowing you to integrate your own vectorization models or indexing strategies.

## Usage

The `Indexer` class provides a simple interface for creating and managing indexes on a Lance dataset.

### Creating an Index

To create an index, you first need to have a Lance dataset. You can create one using the `atlas.sink()` function. Then, you can initialize an `Indexer` with the path to your dataset and use the `create_index()` method.

```python
from atlas.index import Indexer

# Initialize the Indexer with the path to your Lance dataset
idx = Indexer("path/to/sink/operation/output.lance")

# Create a vector index on the 'image' column
# This will automatically generate embeddings for the images
idx.create_index(column="image", index_type="vector")

# Create an FTS index on a 'text' column
idx.create_index(column="text", index_type="fts")
```

### Listing Indexes

You can list the existing indexes on a table to see which columns are indexed and what type of index is being used.

```python
idx.list_indexes()
```

This will print a table with the column names, data types, and index types, similar to the example in the "Core Operations" section.

---
# Analyse

**Coming Soon...**
---
# Training

**Coming Soon...**
