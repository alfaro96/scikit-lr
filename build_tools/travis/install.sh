#!/bin/bash

# This script is meant to be called by the "install" step defined in
# ".travis.yml". See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the ".travis.yml" in the top level folder of the project

# Build the binaries in the same directory than the source code
make inplace
