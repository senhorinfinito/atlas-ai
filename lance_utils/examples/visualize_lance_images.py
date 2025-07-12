import lance
import pyarrow as pa
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys


def visualize_lance_images(dataset_path, num_rows=8):
    ds = lance.dataset(dataset_path)
    table = ds.to_table(columns=["image", "filename"], limit=num_rows)
    images = table.column("image")
    filenames = table.column("filename") if "filename" in table.column_names else [f"Image {i+1}" for i in range(len(images))]

    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axs = axs.flatten()

    for i in range(n):
        img_bytes = images[i].as_py()
        if img_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            axs[i].imshow(img)
            axs[i].set_title(str(filenames[i].as_py()) if filenames else f"Image {i+1}")
            axs[i].axis("off")
        else:
            axs[i].text(0.5, 0.5, "No Image", ha='center', va='center')
            axs[i].axis("off")
    for j in range(n, len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_lance_images.py <lance_dataset_path> [num_rows]")
        sys.exit(1)
    dataset_path = sys.argv[1]
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    visualize_lance_images(dataset_path, num_rows) 