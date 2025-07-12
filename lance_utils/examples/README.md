# Lance Utils Examples

This directory contains example scripts that demonstrate how to use the `lance_utils` library to convert various public datasets into the Lance format.

## How to Run the Examples

1.  **Install the library:**
    Make sure you have installed the `lance_utils` library and its dependencies. From the parent `lance_utils` directory, run:
    ```bash
    pip install .
    ```

2.  **Run an example script:**
    Navigate into this `examples` directory and run any of the scripts directly. For example:
    ```bash
    python convert_object_detection.py
    ```

Each script is self-contained. It will:
- Create a `temp_data` directory for its downloads.
- Download the necessary raw data.
- Perform any necessary pre-processing steps.
- Use `lance_utils` to convert the data into a `.lance` dataset in the current directory.
- Clean up the downloaded files.
