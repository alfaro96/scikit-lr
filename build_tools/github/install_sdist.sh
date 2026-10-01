#!/bin/bash

set -e

python -m pip install pytest

# Build in an isolated environment, as pip does for the users without a wheel
python -m pip install --verbose "$DIST_DIR"/*.tar.gz
