#!/bin/bash

# This script is meant to be called by
# the "test" step defined in cron.yml

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

# Get into a temporal directory
# to run the tests from the
# installed scikit-lr and check
# if we do not leave artifacts
mkdir -p $TEST_DIR

# Copy the setup.cfg file
# for the pytest settings
cp setup.cfg $TEST_DIR
cd $TEST_DIR

# Initialize the test command
TEST_CMD="pytest --showlocals --durations=20 --pyargs"

# Include the coverage in the test
TEST_CMD="$TEST_CMD --cov=sklr"

# Set the format of the report to XML
TEST_CMD="$TEST_CMD --cov-report=xml"

# Include deprecation warnings
# and future warnings in the test
TEST_CMD="$TEST_CMD -Werror::DeprecationWarning"
TEST_CMD="$TEST_CMD -Werror::FutureWarning"

# Print executed commands to the terminal
set -x

# Test scikit-lr
$TEST_CMD sklr
