#!/bin/bash

# This script is meant to be called by the "Execute tests" step defined
# in cron.yml. The behaviour of the script is controlled by the named step
# defined in the cron.yml in the folder .github/workflows of the project.

# Exit immediately if a command
# exits with a non-zero status
set -e

# Ensure that the package versions used to test
# are the same than the installed previously
python --version
python -c 'import numpy; print("NumPy {}".format(numpy.__version__))'
python -c 'import scipy; print("SciPy {}".format(scipy.__version__))'

# Set the command to test the package. In particular, it
# is important to set the format of the report to XML for
# being properly parsed by CodeCov. Also, raise errors on
# deprecation and future warnings to anticipate them
# before stable versions are released
TEST_CMD="pytest --showlocals --durations=20 --pyargs"
TEST_CMD="$TEST_CMD --cov=sklr"
TEST_CMD="$TEST_CMD --cov-report=xml"
TEST_CMD="$TEST_CMD -Werror::DeprecationWarning"
TEST_CMD="$TEST_CMD -Werror::FutureWarning"

# Run the tests
$TEST_CMD sklr
