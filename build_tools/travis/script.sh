#!/bin/bash

# This script is meant to be called by the "script" step defined in
# ".travis.yml". See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the ".travis.yml" in the top level folder of the project

# In this project, the test of the code and the corresponding coverage are executed
make test
