#!/bin/bash

set -e

TEST_CMD="pytest --showlocals --durations=20 --pyargs"
TEST_CMD="$TEST_CMD --cov=sklr --cov-report=xml"
TEST_CMD="$TEST_CMD -n2"

# Run the tests on the installed development version
mkdir tmp_for_test
cp setup.cfg tmp_for_test
cd tmp_for_test

$TEST_CMD sklr
