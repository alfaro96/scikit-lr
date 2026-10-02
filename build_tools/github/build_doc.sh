#!/bin/bash

set -e

python -c "import sklr; sklr.show_versions()"

make -C doc html
