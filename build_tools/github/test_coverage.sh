#!/bin/bash

set -e

python -c "import sklr; sklr.show_versions()"

if [[ "$LINETRACE" == "true" ]]; then
    python build_tools/github/copy_cython_sources.py
fi

python -m coverage run -m pytest
python -m coverage xml
