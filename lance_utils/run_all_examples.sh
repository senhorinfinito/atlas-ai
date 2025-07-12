#!/bin/bash

set -e

echo "--- Setting up virtual environment ---"
python3 -m venv .venv
source .venv/bin/activate

echo "--- Installing dependencies ---"
pip install . pyyaml psutil torch torchvision

echo "--- Running Object Detection Example ---"
python examples/convert_object_detection.py
echo ""

echo "--- All examples ran successfully! ---"
