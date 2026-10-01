#!/bin/bash

set -e

python -m pip install packaging pyyaml

python build_tools/github/check_release.py "$DIST_DIR"
