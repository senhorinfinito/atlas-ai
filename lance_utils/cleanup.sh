#!/bin/bash

echo "--- Cleaning up project directory ---"

# Remove generated lance datasets from the root
rm -rf *.lance

# Remove the temp_data directory from the examples
rm -rf examples/temp_data

# Remove the outputs directory
rm -rf outputs

# Remove the virtual environment
rm -rf .venv

# Remove dummy_data from tests
rm -rf dummy_data

echo "Cleanup complete."
