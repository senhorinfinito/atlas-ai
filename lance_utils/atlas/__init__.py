# lance_utils/__init__.py

"""
A library for discovering annotations in various formats and building flexible Lance datasets.
"""

from .builder import LanceBuilder

def build(input_path: str, output_path: str, batch_size: int = 0, max_rows_per_file: int = 1024):
    """
    Builds a Lance dataset from a source directory by auto-detecting all supported annotations.

    Args:
        input_path (str): Path to the source data directory.
        output_path (str): Path to save the Lance dataset.
        batch_size (int, optional): Batch size for conversion. Defaults to 0 (auto-inferred).
        max_rows_per_file (int, optional): Max rows per file in the Lance dataset. Defaults to 1024.
    """
    builder = LanceBuilder(
        input_path=input_path,
        output_path=output_path,
        batch_size=batch_size,
        max_rows_per_file=max_rows_per_file
    )
    builder.convert()

__version__ = "0.1.0"