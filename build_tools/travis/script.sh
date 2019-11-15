#!/bin/bash

# This script is meant to be called by the "script" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.

# The behavior of the script is controlled by environment variabled
# defined in the .travis.yml in the top level folder of the project.

# Exit immediately if a command exits with a non-zero status
set -e

# Print the Python, NumPy and Scipy versions
python --version
python -c 'import numpy; print("NumPy {}".format(numpy.__version__))'
python -c 'import scipy; print("SciPy {}".format(scipy.__version__))'

# Initialize the test command
TEST_CMD="pytest --showlocals --durations=20 --pyargs"

# Get into a temporal directory to run the tests from the
# installed scikit-lr and check if we do not leave artifacts
mkdir -p $TEST_DIR

# Copy the setup.cfg file for the pytest settings
cp setup.cfg $TEST_DIR
cd $TEST_DIR

# Include the coverage to the test
if [[ "$COVERAGE" == "true" ]]; then
    TEST_CMD="$TEST_CMD --cov sklr"
fi

# Print executed commands to the terminal
set -x

# Test scikit-lr
$TEST_CMD sklr
