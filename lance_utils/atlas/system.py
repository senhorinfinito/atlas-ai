# lance_utils/system.py
import psutil
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

def get_available_memory():
    """Returns available system memory in bytes."""
    return psutil.virtual_memory().available

def estimate_image_memory_footprint(image_paths, sample_size=100):
    """Estimates the average memory footprint of an image."""
    if not image_paths:
        return 10 * 1024 * 1024  # Default to 10MB if no images found
    
    sample_paths = image_paths[:sample_size]
    total_size = 0
    for path in sample_paths:
        try:
            with Image.open(path) as img:
                # Estimate in-memory size as width * height * channels * bytes_per_channel
                total_size += img.width * img.height * len(img.getbands()) * 4
        except Exception:
            # If we can't open it, use file size as a fallback
            total_size += path.stat().st_size
            
    return total_size / len(sample_paths)

def estimate_csv_row_memory_footprint(csv_path, sample_size=1000):
    """Estimates the average memory footprint of a row in a CSV file."""
    try:
        df_sample = pd.read_csv(csv_path, nrows=sample_size)
        return df_sample.memory_usage(deep=True).sum() / len(df_sample)
    except Exception:
        return 1024 # Default to 1KB

def get_auto_batch_size(item_footprint, safety_factor=0.1):
    """
    Calculates a batch size that should fit comfortably in memory.
    
    :param item_footprint: Estimated memory size of a single item in bytes.
    :param safety_factor: Fraction of available memory to target.
    """
    available_memory = get_available_memory()
    target_memory = available_memory * safety_factor
    
    if item_footprint == 0:
        item_footprint = 1 # Avoid division by zero
        
    batch_size = int(target_memory / item_footprint)
    
    return max(1, batch_size)
