#!/bin/bash

set -e

TEST_CMD="pytest --showlocals --durations=20 --pyargs"
TEST_CMD="$TEST_CMD --cov=sklr"
TEST_CMD="$TEST_CMD --cov-report=xml"

$TEST_CMD sklr
