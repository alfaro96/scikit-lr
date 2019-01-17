#!/bin/bash

# This script is meant to be called by the "after_success" step defined in
# ".travis.yml". See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the ".travis.yml" in the top level folder of the project

# In this case, the code coverage is uploaded to codecov plataform
# Codecov is not properly working, avoid this process until fixed
# codecov || echo "codecov upload failed."
