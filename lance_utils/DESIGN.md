# Lance Utils Design Document

This document outlines the technical architecture and design philosophy of the `lance_utils` library.

## 1. Core Philosophy: Annotation-First

The fundamental design principle of this library is **"Annotation-First"**. This is a departure from traditional dataset converters that are often built around rigid, predefined "tasks" (e.g., an "object detection converter").

The limitations of a task-based approach are:
- **Inflexibility:** Real-world datasets are often messy and multi-modal. A single dataset might contain annotations for object detection, instance segmentation, and image captions. A task-based converter forces the user to choose one, losing valuable information.
- **Poor Scalability:** Adding support for a new task or a new data format often requires significant changes to the core conversion logic. Adding support for preference-tuning data (like DPO) or reranking data would be difficult in a vision-centric, task-based system.

The Annotation-First approach solves this by focusing on a more fundamental question: **"What annotations can we find and extract from this data source?"**

The library makes no assumptions about the user's ultimate goal. It simply inspects the source directory, finds all the supported annotations it can, and builds a Lance dataset that represents the complete set of information available.

## 2. Architecture Overview

The conversion process follows a composable pipeline model:

**`Source Data -> (Extractor 1, Extractor 2, ...) -> [DataItem] -> LanceBuilder -> Dynamic Schema -> Lance Dataset`**

1.  **Source Data:** A local directory containing any mix of supported data files (images, text files, JSON, YAML, etc.).
2.  **Extractors:** A suite of small, single-responsibility plugins that each know how to parse one specific type of annotation from the source data.
3.  **DataItem Collection:** The `LanceBuilder` aggregates all extracted annotations, associating them with a generic `DataItem` object. A `DataItem` can represent an image, a piece of text, or a combination of both.
4.  **Dynamic Schema Building:** The builder inspects the types of all collected annotations and dynamically constructs a PyArrow schema. Each unique annotation type gets its own column in the final dataset.
5.  **Lance Dataset:** The builder writes the data, row by row, into a Lance file using the dynamically generated schema.

## 3. Key Components

### `DataItem` (`tasks/base.py`)

This is the central, generic container for a single data point. It is intentionally simple and can hold:
- An `item_id` (e.g., a filename or a generated ID).
- An optional `image_path`.
- Optional `text_content`.
- A dictionary to hold any number of `Annotation` lists, keyed by the annotation type.

### `Annotation` (`tasks/base.py`)

This is the abstract base class for all annotation types. It is the most critical abstraction in the library. A class that inherits from `Annotation` must implement two methods:
1.  `arrow_field(cls) -> pa.Field`: A class method that returns the PyArrow `Field` that defines how this annotation should be represented in the Lance schema. This allows each annotation to be self-describing.
2.  `to_dict(self) -> Dict`: An instance method that returns a dictionary representation of the annotation, which is required for PyArrow's `from_pylist` method.

### `Extractor` (`tasks/base.py`)

This is the abstract base class for all extractors. An `Extractor` is a stateless plugin with a single responsibility: to find and parse one specific type of annotation. It must implement one method:
1.  `extract(self, input_path: Path) -> Dict[str, List[Annotation]]`: This method takes the path to the source data directory and returns a dictionary mapping a data item's ID (e.g., `img1.png` or `doc1.txt`) to a list of the `Annotation` objects it found for that item.

### `LanceBuilder` (`builder.py`)

This is the main orchestrator of the conversion process. Its key responsibilities are:
1.  **Extractor Discovery:** It uses `pkgutil.walk_packages` to automatically find all classes that inherit from `Extractor` within the `atlas.tasks` subpackage. This means that simply adding a new extractor file is enough to register it with the system.
2.  **Annotation Aggregation:** It runs all discovered extractors and intelligently aggregates the results into a collection of `DataItem` objects.
3.  **Schema Generation:** It calls the `arrow_field()` method on each unique `Annotation` type it found to dynamically build the final schema.
4.  **Data Writing:** It iterates through the `DataItem`s, constructs rows that conform to the dynamic schema, and writes them to the Lance dataset in memory-efficient batches.

## 4. Extensibility

The primary goal of this design is to make the library easy to extend.

### Adding a New Annotation Type

Let's say you want to add support for **Keypoints**.

1.  **Create a new `Annotation` subclass:**
    ```python
    # in a new file, e.g., lance_utils/tasks/pose_estimation/extractors.py
    from atlas.tasks.base import Annotation
    import pyarrow as pa

    class Keypoint(Annotation):
        points: List[float]
        visibility: List[int]

        @classmethod
        def arrow_field(cls) -> pa.Field:
            return pa.field("keypoints", pa.list_(pa.struct([
                pa.field("points", pa.list_(pa.float32())),
                pa.field("visibility", pa.list_(pa.int8()))
            ])))

        def to_dict(self) -> Dict[str, Any]:
            return {"points": self.points, "visibility": self.visibility}
    ```

2.  **Create a new `Extractor` subclass:**
    ```python
    # in the same file
    from atlas.tasks.base import Extractor
    import json

    class MyKeypointFormatExtractor(Extractor):
        def extract(self, input_path: Path) -> Dict[str, List[Annotation]]:
            # Your logic to find and parse your specific keypoint format
            # (e.g., reading a custom JSON file)
            output = {}
            # ... find keypoints.json ...
            data = json.load(f)
            for item in data:
                filename = item['image_filename']
                keypoints = Keypoint(...)
                output.setdefault(filename, []).append(keypoints)
            return output
    ```

That's it. The `LanceBuilder` will automatically discover your new extractor, run it, see the new `Keypoint` annotation, and add a `keypoints` column to the Lance dataset with the correct schema. No changes to the core builder or CLI are needed.
