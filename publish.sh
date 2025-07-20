#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Print commands and their arguments as they are executed.
set -x

# 1. Remove old build artifacts
rm -rf build dist atlas_ai.egg-info

# 2. Build the source distribution and wheel
python3 setup.py sdist bdist_wheel

# 3. Upload to PyPI
# The user will be prompted for their username and password here.
python3 -m twine upload dist/*
