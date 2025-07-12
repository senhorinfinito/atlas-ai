# lance_utils/cli.py
import click
from .builder import LanceBuilder
from .visualizer import LanceVisualizer

@click.group()
def cli():
    """A CLI for building, inspecting, and visualizing Lance datasets."""
    pass

@cli.command()
@click.option("-i", "--input-path", required=True, help="Path to the source data directory.")
@click.option("-o", "--output-path", required=True, help="Path to save the Lance dataset.")
@click.option("-b", "--batch-size", default=0, type=int, help="Batch size for conversion. Auto-inferred if not set.")
@click.option("--max-rows-per-file", default=1024, type=int, help="Max rows per file in the Lance dataset.")
def build(input_path: str, output_path: str, batch_size: int, max_rows_per_file: int):
    """Builds a Lance dataset from a source directory by auto-detecting all supported annotations."""
    builder = LanceBuilder(
        input_path=input_path,
        output_path=output_path,
        batch_size=batch_size,
        max_rows_per_file=max_rows_per_file
    )
    builder.convert()

@cli.command()
@click.option("-i", "--input-path", required=True, help="Path to the Lance dataset.")
def inspect(input_path: str):
    """Inspects a Lance dataset and prints its schema."""
    visualizer = LanceVisualizer(input_path)
    visualizer.inspect()

@cli.command()
@click.option("-i", "--input-path", required=True, help="Path to the Lance dataset.")
@click.option("-n", "--num-rows", default=10, type=int, help="Number of rows to visualize.")
def visualize(input_path: str, num_rows: int):
    """Visualizes a sample of a Lance dataset."""
    visualizer = LanceVisualizer(input_path)
    visualizer.visualize(num_rows)

if __name__ == "__main__":
    cli()