#!/bin/bash

set -e

python -m pip install build twine

python -m build --sdist --outdir "$DIST_DIR"

# Check that PyPI will accept the metadata and render the README
python -m twine check --strict "$DIST_DIR"/*.tar.gz
