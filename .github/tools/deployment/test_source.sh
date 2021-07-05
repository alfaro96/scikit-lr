#!/bin/bash

set -e

python -m venv test_env
source test_env/bin/activate

python -m pip install dist/*.tar.gz
python -m pip install pytest

# Run the tests on the source distribution
mkdir tmp_for_test
cp setup.cfg tmp_for_test
cd tmp_for_test

pytest --pyargs sklr
