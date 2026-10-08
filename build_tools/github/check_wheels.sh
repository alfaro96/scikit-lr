#!/bin/bash

set -e

python -m pip install twine

# Check that PyPI will accept the metadata and render the README
python -m twine check --strict "$WHEELS_DIR"/*.whl

python build_tools/github/check_wheel_contents.py "$WHEELS_DIR"/*.whl
