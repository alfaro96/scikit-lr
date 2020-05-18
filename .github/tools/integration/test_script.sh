#!/bin/bash

set -e

python --version
python -c "import numpy; print('NumPy {0}'.format(numpy.__version__))"
python -c "import scipy; print('SciPy {0}'.format(scipy.__version__))"

# Use XML format to upload the code coverage report
# because is the one supported by the GitHub Action
TEST_CMD="pytest --showlocals --durations=20 --pyargs"
TEST_CMD="$TEST_CMD --cov=sklr"
TEST_CMD="$TEST_CMD --cov-report=xml"

$TEST_CMD sklr
