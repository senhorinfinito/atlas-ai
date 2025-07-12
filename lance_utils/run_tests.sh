#!/bin/bash

set -e

echo "--- Setting up virtual environment ---"
python3 -m venv .venv
source .venv/bin/activate

echo "--- Installing dependencies ---"
pip install . pyyaml psutil

echo "--- Cleaning up previous run ---"
rm -rf dummy_data *.lance

echo "--- Creating dummy data ---"

# Create a unified directory for a complex dataset
mkdir -p dummy_data/my_complex_dataset

# Create dummy data files
cat << EOF > dummy_data/my_complex_dataset/pretraining.txt
This is a document for pre-training.
It contains raw text.
EOF

cat << EOF > dummy_data/my_complex_dataset/preferences.jsonl
{"prompt": "Why is the sky blue?", "chosen": "Rayleigh scattering.", "rejected": "It's painted."}
EOF

echo "Dummy data created."
echo ""

mkdir -p outputs

# --- Run Tests ---

echo "--- Testing auto-detection of all tasks ---"
atlas build -i dummy_data/my_complex_dataset -o outputs/all_annotations.lance
atlas inspect -i outputs/all_annotations.lance
echo "--> Note: Schema should contain item_id, text, and preference."
echo ""

echo "--- All tests passed successfully! ---"