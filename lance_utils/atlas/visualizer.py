# atlas/visualizer.py
import lance
import pyarrow as pa
from rich.console import Console
from rich.table import Table
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import math
from pathlib import Path

class LanceVisualizer:
    """Visualizes and inspects Lance datasets."""

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.dataset = lance.dataset(self.input_path)
        self.console = Console()

    def inspect(self):
        """Prints the dataset schema and metadata."""
        schema = self.dataset.schema
        
        table = Table(title=f"Schema for {self.input_path}")
        table.add_column("Field Name", justify="right", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")

        for field in schema:
            table.add_row(field.name, str(field.type))
            
        self.console.print(table)
        self.console.print(f"Number of rows: {self.dataset.count_rows()}")

    def visualize(self, num_rows: int = 10):
        """Visualizes a sample of the dataset."""
        if "image" in self.dataset.schema.names:
            self._visualize_images(num_rows)
        else:
            self._visualize_tabular(num_rows)

    def _visualize_tabular(self, num_rows: int):
        """Visualizes tabular data."""
        df = self.dataset.to_table(limit=num_rows).to_pandas()
        
        table = Table(title="Tabular Data Sample")
        for col in df.columns:
            table.add_column(col, style="cyan")
        
        for _, row in df.iterrows():
            table.add_row(*[str(item) for item in row])
            
        self.console.print(table)

    def _visualize_images(self, num_rows: int):
        """Saves a visualization of the image data and its annotations to a file."""
        sample = self.dataset.to_table(limit=num_rows).to_pylist()
        
        if not sample:
            self.console.print("No data to display.")
            return

        num_cols = 4
        num_rows_plot = math.ceil(len(sample) / num_cols)
        fig, axs = plt.subplots(num_rows_plot, num_cols, figsize=(15, 4 * num_rows_plot), squeeze=False)
        
        for i, item in enumerate(sample):
            row = i // num_cols
            col = i % num_cols
            ax = axs[row, col]

            image_bytes = item.get("image")
            if not image_bytes:
                ax.text(0.5, 0.5, "No Image", ha='center')
                ax.axis("off")
                continue

            try:
                image = Image.open(io.BytesIO(image_bytes))
                ax.imshow(image)
                title = item.get("filename") or f"Image {i+1}"
                ax.set_title(title, fontsize=8)
                
                # --- Annotation Drawing Logic ---
                if "bounding_boxes" in item and item["bounding_boxes"]:
                    for ann in item["bounding_boxes"]:
                        box = ann['bbox']
                        rect = patches.Rectangle(
                            (box[0], box[1]), box[2], box[3],
                            linewidth=1, edgecolor='r', facecolor='none'
                        )
                        ax.add_patch(rect)
                        if 'category_id' in ann:
                            ax.text(box[0], box[1] - 5, str(ann['category_id']), 
                                    color='white', backgroundcolor='red', fontsize=6)

                if "caption" in item and item["caption"]:
                     fig.text(col/num_cols, (num_rows_plot-row-1)/num_rows_plot - 0.05, item["caption"], 
                              fontsize=7, wrap=True, ha='center')

            except Exception as e:
                ax.text(0.5, 0.5, f"Could not load image\n{e}", ha='center')
            
            ax.axis("off")

        for i in range(len(sample), num_cols * num_rows_plot):
            row = i // num_cols
            col = i % num_cols
            axs[row, col].axis("off")

        plt.tight_layout(pad=2.0)
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "visualization.png"
        
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free memory
        
        self.console.print(f"Visualization saved to [bold cyan]{output_path}[/bold cyan]")