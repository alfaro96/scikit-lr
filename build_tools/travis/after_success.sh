#!/bin/bash

# This script is meant to be called by the "after_success" step defined
# in .travis.yml. See https://docs.travis-ci.com/ for more details.

# Exit immediately if a command exits with a non-zero status
set -e

# Need to run codecov from a git checkout, so copy
# .coverage from TEST_DIR where pytest has been run
cp $TEST_DIR/.coverage $TRAVIS_BUILD_DIR

# Ignore codecov failures as the codecov server is not very
# reliable but it is not wanted that Travis reports a failure
# in the Github UI just because the coverage report failed to
# be published
codecov --root $TRAVIS_BUILD_DIR || echo "codecov upload failed"
