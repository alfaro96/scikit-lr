#!/bin/bash

set -e

python -m venv build_env
source build_env/bin/activate

pip install numpy scipy cython
pip install twine

python setup.py sdist

# Check the rendering of the source distribution
twine check dist/*.tar.gz
