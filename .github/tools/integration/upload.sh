#!/bin/bash

# This script is meant to be called by the
# "upload" step defined in integration.yml

# Exit immediately if a command exits with a non-zero status
set -e

# Need to run codecov from a git checkout,
# so copy .coverage from TEST_DIR where
# pytest has been run
cp $TEST_DIR/.coverage $GITHUB_WORKSPACE

# Ignore codecov failures as the codecov server is not very
# reliable but it is not wanted that Github Actions reports
# a failure in the Github UI just because the coverage report
# has failed to be published
codecov --root $GITHUB_WORKSPACE || echo "codecov upload failed"
