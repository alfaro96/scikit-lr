#!/bin/bash

# This script is meant to be called by the
# "test" step defined in integration.yml

# The behavior of the script is controlled by
# the named step defined in the scheduled.yml
# in the folder .github/workflows of the project

# Exit immediately if a command
# exits with a non-zero status
set -e

# Print the Python, NumPy and Scipy versions
python --version
python -c 'import numpy; print("NumPy {}".format(numpy.__version__))'
python -c 'import scipy; print("SciPy {}".format(scipy.__version__))'

# Initialize the test command
TEST_CMD="pytest --showlocals --durations=20 --pyargs"

# Include the coverage in the test
TEST_CMD="$TEST_CMD --cov=sklr"

# Set the format of the report to XML
TEST_CMD="$TEST_CMD --cov-report=xml"

# Print executed commands to the terminal
set -x

# Test scikit-lr
$TEST_CMD sklr
