#!/bin/bash

set -e

python -m venv test_env
source test_env/bin/activate

pip install dist/*.tar.gz
pip install pytest

# Run the tests on the installed source distribution
mkdir tmp_for_test
cp setup.cfg tmp_for_test
cd tmp_for_test

pytest --pyargs sklr
